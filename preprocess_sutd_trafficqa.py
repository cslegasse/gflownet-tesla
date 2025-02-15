from pathlib import Path
import os
import json
import pandas as pd

ALL_INPUT_FILE = (
    "sutd_trafficqa/annotations/R2_all.jsonl"
)
ALL_OUTPUT_DIR = "data/sutd_trafficqa/all"
TEST_INPUT_FILE = (
    "sutd_trafficqa/annotations/R2_test.jsonl"
)
TEST_OUTPUT_DIR = "data/sutd_trafficqa/test"
TRAIN_INPUT_FILE = (
    "sutd_trafficqa/annotations/R2_train.jsonl"
)
TRAIN_OUTPUT_DIR = "data/sutd_trafficqa/train"
DEFAULT_INPUT_FILE = ALL_INPUT_FILE
DEFAULT_OUTPUT_DIR = ALL_OUTPUT_DIR


def main(input_file: str = DEFAULT_INPUT_FILE, output_dir: str = DEFAULT_OUTPUT_DIR):
    annotation_file = input_file
    questions = []
    answers = []
    id_counter = 1

    with open(annotation_file, "r") as f:
        lines = f.readlines()

    _header = lines.pop(0)

    Q_TYPE_MAP = {
        "U": "Basic Understanding",
        "A": "Attribution",
        "F": "Event Forecasting",
        "R": "Reverse Reasoning",
        "C": "Counterfactual Inference",
        "I": "Introspection",
    }

    for line in lines:
        data = json.loads(line.strip())

        # Unique ID of this data point
        record_id = data[0]
        # Unique ID of the source video
        vid_id = data[1]
        # File name of the source video
        vid_filename = data[2]
        # 1 or 3 denotes first-person or third-person perspective
        perspective = data[3]
        q_body = data[4]
        q_type = data[5]
        option0 = data[6]
        option1 = data[7]
        option2 = data[8]
        option3 = data[9]
        answer_idx = data[10]

        q_type = Q_TYPE_MAP.get(q_type, "Unknown")
        options = [option0, option1, option2, option3]
        options_str = (
            " A. " + option0 + " B. " + option1 + " C. " + option2 + " D. " + option3
        )
        question_str = q_body + options_str

        questions.append(
            {"id": f"{id_counter:05d}", "question": question_str, "vid_filename": f"{vid_filename}"},
        )
        answers.append(
            {
                "id": f"{id_counter:05d}",
                "answer": chr(65 + answer_idx),
                "vid_filename": f"{vid_filename}",
            }
        )  # Convert index to A, B, C, D

        id_counter += 1

    # Create DataFrames
    questions_df = pd.DataFrame(questions)
    answers_df = pd.DataFrame(answers)

    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    questions_df.to_csv(Path(output_dir) / "questions.csv", index=False)
    answers_df.to_csv(Path(output_dir) / "answers.csv", index=False)

    print(
        f"Preprocessing complete. questions.csv and answers.csv created in {output_dir}"
    )


if __name__ == "__main__":
    main(input_file=DEFAULT_INPUT_FILE, output_dir=DEFAULT_OUTPUT_DIR)
    main(input_file=TEST_INPUT_FILE, output_dir=TEST_OUTPUT_DIR)
    main(input_file=TRAIN_INPUT_FILE, output_dir=TRAIN_OUTPUT_DIR)
