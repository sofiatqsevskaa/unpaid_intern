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
    print("results:")

    for i, res in enumerate(result.get("results", []), 1):
        print(f"{i}. {res.get('metadata', {}).get('document_name', 'unknown')}")
        content = res.get("content", "")
        print(f"content: {content[:150]}...")
        print(f"distance: {res.get('distance', 0):.4f}")

    print("thats all folks from test vector query\n")

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
    print("results:")
    for i, res in enumerate(result.get("results", []), 1):
        doc = res.get("document", {})
        print(f"{i}. {doc.get('name', 'unknown')}")
        print(f"preview: {doc.get('content_preview', '')[:150]}...")
        entities = res.get("entities", [])
        if entities:
            formatted = [
                f"{e.get('name')} ({e.get('type')})" for e in entities]
            print(f"entities: {', '.join(formatted)}")

    print("thats all folks from test graph query\n")
    return result


if __name__ == "__main__":
    print("TESTING VECTOR QUERIES  \n")

    test_vector_query("Where did Sara go?")
    test_vector_query("What did Sara buy at the market?")
    test_vector_query("How much did Sara spend?")
    test_vector_query("Where is the farmers market located?")

    test_vector_query("What did Michael buy?")
    test_vector_query("Where is Leo's Electronics Repair Shop?")
    test_vector_query("How much did Michael pay for the synthesizer?")
    test_vector_query("What did Michael do with the synthesizer?")

    test_vector_query("What kind of tree did Chloe buy?")
    test_vector_query("Where does Chloe live?")
    test_vector_query("How often does Chloe water her tree?")
    test_vector_query("Who helped Chloe at the garden center?")

    test_vector_query("people who bought things")
    test_vector_query("places in the stories")
    test_vector_query("prices and costs")

    print("\nTESTING GRAPH QUERIES\n")

    test_graph_query("Sara")
    test_graph_query("Michael")
    test_graph_query("Chloe")
    test_graph_query("Leo")
    test_graph_query("Martha")

    test_graph_query("farmers market")
    test_graph_query("repair shop")
    test_graph_query("garden center")
    test_graph_query("Main Street")
    test_graph_query("Central Park")

    test_graph_query("tomatoes")
    test_graph_query("synthesizer")
    test_graph_query("maple sapling")
    test_graph_query("mug")
    test_graph_query("honey")

    test_graph_query("bought something")
    test_graph_query("planted a tree")
    test_graph_query("visited a shop")
    test_graph_query("made tomato sauce")

    print("ALL TESTS COMPLETE")
