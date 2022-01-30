import requests


def query_api(url, data):
    output = requests.post(url, json=data)
    if output.status_code == 200:
        return output.status_code, output.json()
    return output.status_code, None


if __name__ == '__main__':
    data = {
        "age": 28,
        "workclass": "Private",
        "fnlwgt": 338409,
        "education": "Bachelors",
        "education_num": 20,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "India"
    }

    status_code, output = query_api("https://fastapi-inference.herokuapp.com/inference", data)

    print("STATUS CODE:", status_code)
    print("OUTPUT:", output)
