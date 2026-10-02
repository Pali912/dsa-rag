import json

with open("eval_checkpoint.json", encoding="utf-8") as f:
    checkpoint = json.load(f)

with open("all_answers.txt", "w", encoding="utf-8") as out:
    for question, data in checkpoint.items():
        out.write("=" * 60 + "\n")
        out.write("Q: " + question + "\n")
        out.write("-" * 60 + "\n")
        out.write(data["answer"] + "\n\n")

print("Written to all_answers.txt")
