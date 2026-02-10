import requests
import traceback

base_url = "http://localhost:8000"
user_id = "test_user_001"


def print_error(response=None, exception=None):
    print("error occurred")
    if exception:
        print(f"exception type: {type(exception).__name__}")
        print(f"exception message: {exception}")
        traceback.print_exc()
    if response is not None:
        print(f"http status: {response.status_code}")
        try:
            print("response json:", response.json())
        except Exception:
            print("response text:", response.text)
    print()


def safe_request(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
        if not response.ok:
            print_error(response=response)
            return None
        try:
            return response.json()
        except Exception as e:
            print_error(response=response, exception=e)
            return None
    except Exception as e:
        print_error(exception=e)
        return None


def test_vector_query(query):
    print(f"vector query: {query}")

    result = safe_request(
        "get",
        f"{base_url}/query/vector",
        params={"user_id": user_id, "query": query, "top_k": 3}
    )

    if result is None:
        return None

    print(f"found {result.get('results_count', 0)} results")

    for i, res in enumerate(result.get("results", []), 1):
        print(f"{i}. {res.get('metadata', {}).get('document_name', 'unknown')}")
        content = res.get("content", "")
        print(f"content: {content[:150]}...")
        print(f"distance: {res.get('distance', 0):.4f}")

    print()
    return result


def test_graph_query(query):
    print(f"graph query: {query}")

    result = safe_request(
        "get",
        f"{base_url}/query/graph",
        params={"user_id": user_id, "query": query}
    )

    if result is None:
        return None

    print(f"found {result.get('results_count', 0)} results")

    for i, res in enumerate(result.get("results", []), 1):
        doc = res.get("document", {})
        print(f"{i}. {doc.get('name', 'unknown')}")
        print(f"preview: {doc.get('content_preview', '')}")
        entities = res.get("entities", [])
        if entities:
            formatted = [
                f"{e.get('name')} ({e.get('type')})" for e in entities]
            print(f"entities: {', '.join(formatted)}")

    print()
    return result


def test_upload():
    print("testing upload")
    test_content = "where did sara go?"
    try:
        with open("test_upload.txt", "w") as f:
            f.write(test_content)
    except Exception as e:
        print_error(exception=e)
        return None
    try:
        with open("test_upload.txt", "rb") as f:
            result = safe_request(
                "post",
                f"{base_url}/upload",
                data={
                    "user_id": user_id,
                    "document_name": "test_upload.txt",
                    "tags": "test,upload",
                    "description": "test upload document"
                },
                files={"file": f}
            )
    except Exception as e:
        print_error(exception=e)
        return None

    if result is None:
        return None

    for res in result:
        print(f"{res.get('kb_type')}: {res.get('status')}")
        if res.get("chunks_processed"):
            print(f"chunks: {res.get('chunks_processed')}")
        if res.get("entities_extracted"):
            print(f"entities: {res.get('entities_extracted')}")

    print()
    return result


if __name__ == "__main__":
    print("testing system")
    test_vector_query("Where did Sara go?")
    test_graph_query("repair shop")
    test_upload()
    test_vector_query("machine")
