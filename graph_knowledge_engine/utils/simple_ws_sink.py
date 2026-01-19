import asyncio
import websockets
async def m(): 
    async with websockets.connect('ws://localhost:8787/changes/ws') as w: 
        async for x in w: 
            print(x)
asyncio.run(m())
