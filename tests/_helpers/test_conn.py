import httpx
import asyncio
import os

async def test_httpx_conn():
    raw_url = os.getenv("GRAPHRAG_SERVER_URL", "http://localhost:28110")
    print(f"DEBUG: raw_url repr: {repr(raw_url)}")
    url = raw_url.strip().strip('"').strip("'")
    print(f"Testing connectivity to {url}...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Try absolute URL first
            print(f"Calling GET {url}/")
            resp = await client.get(f"{url}/")
            print(f"ROOT: {resp.status_code}")
            
            # Try /api/conversations
            print(f"Calling GET {url}/api/conversations")
            resp = await client.get(f"{url}/api/conversations")
            print(f"/api/conversations: {resp.status_code} (expected 401)")
            
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_httpx_conn())
