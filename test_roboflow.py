from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rF2UaYsyVRscRHH14q4u"
)

# Replace this with the path to a real image file on your system
image_path = "data/soccer_ball/images/train/frame_000000_b2.jpg"

result = client.infer(image_path, model_id="football-players-detection-3zvbc/12")
print(result) 