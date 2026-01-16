import torch
import numpy as np
from terratorch import FULL_MODEL_REGISTRY

# === importy z terramindfunctions - używany jako biblioteka
import terramindFunctions as tm
from metrics import calculate_accuracy, calculate_miou, calculate_final_score, print_report

# Konfiguracja Nauczyciela
TEACHER_MODEL_NAME = "terratorch_terramind_v1_large_generate"
DEVICE = tm.device # Używamy tego samego device co w terramindFunctions

def load_teacher_model():
    """Ładuje model Large (Nauczyciela), którego nie ma w terramindFunctions."""
    print(f"Ładowanie Nauczyciela: {TEACHER_MODEL_NAME}...")
    try:
        model = FULL_MODEL_REGISTRY.build(
            TEACHER_MODEL_NAME,
            modalities=["S2L2A"],
            output_modalities=["LULC"],
            pretrained=True,
            standardize=True,
        ).to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Błąd ładowania Nauczyciela: {e}")
        return None

def run_evaluation(lat, lon, buffer_km=5):
    print(f"ROZPOCZYNAM EWALUACJĘ DLA: {lat}, {lon}")
    
    # 1. POBRANIE DANYCH (Używamy funkcji z terramindFunctions)
    # To gwarantuje, że oba modele dostaną te same dane wejściowe
    dl_result = tm.download_sentinel2(lat, lon, buffer_km, max_cloud_cover=10, days_back=180)
    
    if dl_result is None:
        print("Nie udało się pobrać danych. Koniec testu.")
        return

    raw_data, date, _ = dl_result
    
    # 2. PRZYGOTOWANIE TENSORA (Używamy Twojej funkcji)
    input_tensor = tm.prepare_input(raw_data)
    
    # ==========================================
    # STUDENT (model Small)
    # ==========================================
    print("Generowanie mapy Studenta (Small)...")
    student_model = tm.get_model() # Pobieramy zcache'owany model z Twojego pliku
    student_raw_output = tm.run_inference(student_model, input_tensor)
    student_map = tm.decode_output(student_raw_output) # Zamiana na klasy ESA (10, 20, 80...)

    # ==========================================
    # NAUCZYCIEL (model Large)
    # ==========================================
    print("Generowanie mapy Nauczyciela (Large)...")
    teacher_model = load_teacher_model()
    
    if teacher_model is None:
        return

    with torch.no_grad():
        teacher_out = teacher_model(
            {"S2L2A": input_tensor.to(DEVICE)},
            verbose=False,
            timesteps=tm.TIMESTEPS
        )
        teacher_raw_tensor = teacher_out["LULC"].detach()
    
    # Używamy funkcji decode_output, aby Nauczyciel też zwracał klasy ESA
    teacher_map = tm.decode_output(teacher_raw_tensor)

    # Zwolnij pamięć po dużym modelu
    del teacher_model
    torch.cuda.empty_cache()

    # ==========================================
    # OBLICZANIE METRYK
    # ==========================================
    print("Obliczanie wyników...")
    
    acc = calculate_accuracy(student_map, teacher_map)   
    miou = calculate_miou(student_map, teacher_map)

    final_score = calculate_final_score(acc, miou)
    
    print_report(acc, miou, final_score, model_name="Student (vs Large)")
    
    return {
        "accuracy": acc,
        "miou": miou,
        "final_score": final_score,
        "student_map": student_map,
        "teacher_map": teacher_map
    }



# Przykładowe uruchomienie
if __name__ == "__main__":
    # Współrzędne
    run_evaluation(50.0647, 19.9450, buffer_km=3)   #generalnie tu moduł z metrykami musiałby "zassać" dane z apki jesli chcemy go dorzucić