import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from google import genai
from google.genai.types import HttpOptions, Part


app = FastAPI()
vertexai.init(project='playground-elly-kim-bda1')

MAX_RETRIES=3

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_fitb(image_bytes: bytes) -> str:
    client = genai.Client(
        vertexai=True,
        project="playground-elly-kim-bda1",
        location="us-central1",
        http_options=HttpOptions(api_version="v1")
    )
    
    prompt = """
## Objective
너는 OCR processor야. 원본 이미지의 포맷을 최대한 유지하면서 텍스트를 markdown 형식으로 추출해줘.

## Output Example
```markdown
2강. 구석기와 신석기
A. 구석기
1) 채집, 수렵으로 이동 생활을 했다.
2) 가족을 중심 단위로 생활했다.
3) 동굴, 막집에서 살았으며, 대표적인 유적지로 평양, 전곡리, 공주 석장리 등이 있다.
4) 대표적 도구로 뗀석기를 이용하였다.
5) 무리 사회이자 평등 사회였다.
6) 홍수아이(이곡리 유적)에서 시신 매장 풍습을 볼 수 있다.
```
"""
    
    for i in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[
                    prompt,
                    Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    ),
                ],
            )
        except:
            if (i + 1) == MAX_RETRIES:
                return "FAILED"
    content = response.text.removeprefix("```markdown").removesuffix("```")

    prompt = f"""
## Objective
사용자는 아래의 내용을 외우고 싶어해. 그래서 시험에 출제될 수 있는 중요한 부분에 빈칸을 만들어서 자체적인 시험지를 만들어 줄거야.
아래의 내용을 읽고 출제될 가능성이 높은 중요한 단어에 '**중요한 단어**' 표시를 해줘.

반환문 예시:
## Output Example
```json
{{
"fill_in_the_blank_quiz":"2강. 구석기와 신석기\nA. 구석기\n1) 채집, 수렵으로 **이동** 생활을 했다.\n2) **가족**을 중심 단위로 생활했다.\n3) 동굴, 막집에서 살았으며, 대표적인 유적지로 **평양**, **전곡리**, **공주 석장리** 등이 있다.\n4) 대표적 도구로 **뗀석기**를 이용하였다.\n5) 무리 사회이자 **평등** 사회였다.\n6) 홍수아이(이곡리 유적)에서 **시신 매장** 풍습을 볼 수 있다."
}}
```

유저의 자료:
{content}
"""
    for i in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents=[
                    prompt,
                ],
            )
            quiz = response.text.removeprefix("```json").removesuffix("```")
            return json.loads(quiz)["fill_in_the_blank_quiz"]
        except:
            if (i + 1) == MAX_RETRIES:
                return "FAILED"

@app.post("/quiz")
async def generate_quiz(
    file: UploadFile = File(...)
):
    data = await file.read()
    quiz = generate_fitb(data)
    return {"quiz": quiz}
