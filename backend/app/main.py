from fastapi import FastAPI

app=FastAPI(
    title="AML Investigator Copilot",
    description="AI-assisted financial crime investigation platform",
    version="0.1.0"
)

@app.get("/")
def root():
    return {
        "message":"AML Investigator Copilot is running",
        "status":'healthy'
    }