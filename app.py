from fastapi import FastAPI
import requests
import uvicorn

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8000/render"

app = FastAPI()


@app.post(f"{BACKEND_URL}/draw_card")
def draw_card(card: int):
    requests.post(f"{FRONTEND_URL}/draw_card", json={"card": card})


@app.post(f"{BACKEND_URL}/calculate_hand")
def calculate_hand(cards: List[int]):
    requests.post(f"{FRONTEND_URL}/calculate_hand", json={"cards": cards})

@app.post(f"{BACKEND_URL}/get_observation")
def get_observation(player_cards: List[int], dealer_cards: List[int]):
    requests.post(f"{FRONTEND_URL}/get_observation", json={"player_cards": player_cards, "dealer_cards": dealer_cards})

@app.post(f"{BACKEND_URL}/step")
def step(action: int, reward: float, done: bool, info: Dict):
    requests.post(f"{FRONTEND_URL}/step", json={"action": action, "reward": reward, "done": done, "info": info})

@app.post(f"{BACKEND_URL}/render")
def render(state: Dict):
    requests.post(f"{FRONTEND_URL}/render", json={"state": state})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)