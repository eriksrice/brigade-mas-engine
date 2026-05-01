from fastapi import FastAPI, Request, BackgroundTasks
from brigade import node_courier, BrigadeState  # Importing your brain

app = FastAPI()

@app.post("/webhook")
async def linkedin_webhook(request: Request, background_tasks: BackgroundTasks):
    # 1. Catch the payload from Google Pub/Sub
    envelope = await request.json()
    
    # 2. Package the state specifically for The Courier
    state: BrigadeState = {
        "webhook_data": envelope,
        "target_agent": "courier"
    }
    
    # 3. Hand the execution off to the background thread
    # This calls the node_courier function inside brigade.py
    background_tasks.add_task(node_courier, state)
    
    # 4. Immediately return 200 OK to Google so it closes the loop
    return {"status": "Message received, Courier dispatched"}