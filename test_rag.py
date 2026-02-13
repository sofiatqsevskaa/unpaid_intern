import requests
import traceback
import json

BASE_URL = "http://localhost:8000"
USER_ID = "test_user_001"


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


def query_vector_db(query, top_k=3):
    print(f"VECTOR DB: '{query}'")

    result = safe_request(
        "get",
        f"{BASE_URL}/query/vector",
        params={"user_id": USER_ID, "query": query, "top_k": top_k}
    )

    if result is None:
        return []

    print(f"found {result.get('results_count', 0)} results")
    return result.get("results", [])


def query_graph_db(query):
    print(f"GRAPH DB: '{query}'")

    result = safe_request(
        "get",
        f"{BASE_URL}/query/graph",
        params={"user_id": USER_ID, "query": query}
    )

    if result is None:
        return []

    print(f"found {result.get('results_count', 0)} results")
    return result.get("results", [])


def rag_query(query, use_vector=True, use_graph=True, top_k=3):
    print(f"RAG QUERY: '{query}'")

    vector_results = query_vector_db(query, top_k) if use_vector else []
    graph_results = query_graph_db(query) if use_graph else []

    print(
        f"Sending to LLM with {len(vector_results)} vector + {len(graph_results)} graph results.")

    payload = {
        "query": query,
        "vector_results": vector_results,
        "graph_results": graph_results,
        "max_tokens": 500,
        "temperature": 0.3
    }

    result = safe_request(
        "post",
        f"{BASE_URL}/rag/query",
        json=payload
    )

    if result:
        print(f"ANSWER:")
        print(result.get("response"))
        print(f"Context used: {result.get('context_used', {})}")

    return result


def test_single_queries():
    print("TESTING INDIVIDUAL DATABASES")
    query_vector_db("Where did Sara go?")
    query_vector_db("What did Michael buy?")
    query_vector_db("What kind of tree did Chloe buy?")

    query_graph_db("Sara")
    query_graph_db("synthesizer")
    query_graph_db("maple sapling")


def test_rag_queries():
    print("TESTING RAG WITH BOTH DATABASES")
    rag_query("Where did Sara go on Saturday and what did she buy?")
    rag_query(
        "How much did Sara spend and what did she do with her purchases later?")

    rag_query("What did Michael buy from Leo and how much did it cost?")
    rag_query(
        "Where did Michael take his new synthesizer and what did he do with it?")

    rag_query("What kind of tree did Chloe buy and where did she plant it?")
    rag_query("Who helped Chloe at the garden center and what advice did they give?")
    rag_query("What are the names of all the people in the stories?")
    rag_query("What are the different locations mentioned in all the documents?")
    rag_query("How much money did each person spend?")


def test_vector_only():
    print("TESTING RAG WITH VECTOR ONLY")
    rag_query("What did Sara buy at the market?", use_graph=False)


def test_graph_only():
    print("TESTING RAG WITH GRAPH ONLY")
    rag_query("Who is Leo and what does he do?", use_vector=False)


if __name__ == "__main__":
    health = safe_request("get", f"{BASE_URL}/health")
    if not health:
        print("server not responding.")
        exit(1)

    print("server is healthy. starting tests")
    test_single_queries()
    test_rag_queries()
    test_vector_only()
    test_graph_only()

    print("\nALL TESTS COMPLETE")
