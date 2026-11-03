import json

# Define your ground truth labels
GOLD_STANDARDS = {
    "Sample 1": {
        "tender_no": "MnC/LUB-19/PT-50/26-27",
        "tech_evaluation_mode": "L1",
        "ato_minimum": 117998000.0,
    },
    "Sample 3": {
        "tender_no": "MnC/LUB-13/PT-33/26-27",
        "tech_evaluation_mode": "QCBS",
        "ato_minimum": 5471000.0,
    }
}

def evaluate_extraction(sample_name: str, extracted_file: str):
    with open(extracted_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    truth = GOLD_STANDARDS[sample_name]
    
    extracted_mode = data.get("tech_evaluation_mode", {}).get("mode") if isinstance(data.get("tech_evaluation_mode"), dict) else data.get("tech_evaluation_mode")
    extracted_tender_no = data.get("tender no", {}).get("value") if isinstance(data.get("tender no"), dict) else data.get("tender no")

    mode_match = extracted_mode == truth["tech_evaluation_mode"]
    tender_match = truth["tender_no"] in str(extracted_tender_no)

    print(f"--- Eval Report: {sample_name} ---")
    print(f"Evaluation Mode Match: {'PASS' if mode_match else 'FAIL'} (Expected: {truth['tech_evaluation_mode']}, Got: {extracted_mode})")
    print(f"Tender ID Recall:     {'PASS' if tender_match else 'FAIL'} (Expected contains: {truth['tender_no']})")
    print(f"Evidence Attached:     {'PASS' if data.get('name of work', {}).get('source_quote') else 'FAIL'}")

if __name__ == "__main__":
    evaluate_extraction("Sample 1", "Sample 1_extracted.json")
    evaluate_extraction("Sample 3", "Sample 3_extracted.json")
    evaluate_extraction("Sample 2", "Sample 2_extracted.json")
    