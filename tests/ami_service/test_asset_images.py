from ami_rag.storage.assets import IMAGE_EXTENSIONS, MinioAssetStore


class _Obj:
    def __init__(self, object_name: str):
        self.object_name = object_name


class StubMinioClient:
    """Minio stub: chỉ list_objects + presigned_get_object (không network)."""

    def __init__(self, keys=(), fail_presign_for=()):
        self.keys = list(keys)
        self.fail_presign_for = set(fail_presign_for)

    def list_objects(self, bucket, prefix="", recursive=False):
        return [_Obj(k) for k in self.keys if k.startswith(prefix)]

    def presigned_get_object(self, bucket, key, expires=None):
        if key in self.fail_presign_for:
            raise RuntimeError("presign boom")
        return f"https://minio/{key}"

    def get_object(self, bucket, key):
        raise AssertionError("not expected in these tests")

    def put_object(self, bucket, key, data, length, content_type=None):
        raise AssertionError("not expected in these tests")


def _store(client) -> MinioAssetStore:
    return MinioAssetStore(
        endpoint="m",
        access_key="a",
        secret_key="s",
        bucket="b",
        client=client,
    )


def test_list_doc_images_filters_media_and_sorts():
    keys = [
        "rag-assets/d1/zz.jpg",
        "rag-assets/d1/content_list.json",
        "rag-assets/d1/img.png",
        "rag-assets/d1/descriptions.json",
        "rag-assets/d1/chunks.json",
        "rag-assets/d1/clip.mp4",
        "rag-assets/d1/note.txt",
        "rag-assets/d1/IMG2.PNG",
        "rag-assets/other/img.png",
    ]
    store = _store(StubMinioClient(keys))
    assert store.list_doc_images("d1") == [
        "rag-assets/d1/IMG2.PNG",
        "rag-assets/d1/clip.mp4",
        "rag-assets/d1/img.png",
        "rag-assets/d1/zz.jpg",
    ]


def test_list_doc_images_empty_doc():
    store = _store(StubMinioClient())
    assert store.list_doc_images("missing") == []


def test_list_doc_images_ext_subset_matches_crawl_exts():
    # .webm/.mp4 (crawl “ảnh”/media) và .bmp/.tiff đều nằm trong IMAGE_EXTENSIONS
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".webm", ".mp4"):
        assert ext in IMAGE_EXTENSIONS


def test_presign_doc_images_skips_failures():
    keys = [
        "rag-assets/d1/a.png",
        "rag-assets/d1/content_list.json",
        "rag-assets/d1/b.jpg",
    ]
    store = _store(StubMinioClient(keys, fail_presign_for={"rag-assets/d1/a.png"}))
    urls = store.presign_doc_images("d1")
    assert urls == ["https://minio/rag-assets/d1/b.jpg"]


def test_presign_doc_images_all_fail_or_none():
    store = _store(StubMinioClient(["rag-assets/d1/a.png"], fail_presign_for={"rag-assets/d1/a.png"}))
    assert store.list_doc_images("d1") == ["rag-assets/d1/a.png"]
    assert store.presign_doc_images("d1") == []
