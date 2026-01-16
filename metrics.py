import numpy as np

def calculate_accuracy(pred_map, target_map):
    """
    Oblicza Pixel Accuracy (procent zgodnych pikseli).
    
    Args:
        pred_map (numpy.ndarray): Mapa klas predykcji (Student).
        target_map (numpy.ndarray): Mapa klas wzorca (Nauczyciel).
        
    Returns:
        float: Wynik w procentach (0-100).
    """
    # Spłaszczamy mapy do 1D dla łatwiejszego porównania
    p = pred_map.flatten()
    t = target_map.flatten()
    
    correct_pixels = np.sum(p == t)
    total_pixels = p.size
    
    return (correct_pixels / total_pixels) * 100.0

def calculate_miou(pred_map, target_map):
    """
    Oblicza mIoU (Mean Intersection over Union).
    Ignoruje klasy, które nie występują we wzorcu.
    
    Args:
        pred_map (numpy.ndarray): Mapa klas predykcji.
        target_map (numpy.ndarray): Mapa klas wzorca.
        
    Returns:
        float: Wynik mIoU w procentach (0-100).
    """
    # Unikalne klasy obecne we wzorcu (Ground Truth)
    classes = np.unique(target_map)
    iou_list = []
    
    for cls in classes:
        # Pomiń klasę 0 (Brak danych), jeśli nie chcesz jej oceniać
        if cls == 0:
            continue
            
        # Tworzymy maski binarne dla danej klasy
        p_mask = (pred_map == cls)
        t_mask = (target_map == cls)
        
        intersection = np.logical_and(p_mask, t_mask).sum()
        union = np.logical_or(p_mask, t_mask).sum()
        
        if union > 0:
            iou = intersection / union
            iou_list.append(iou)
    
    if len(iou_list) == 0:
        return 0.0
        
    return np.mean(iou_list) * 100.0

def calculate_final_score(accuracy, miou):
    """
    Oblicza ostateczny wynik ważony wg wzoru:
    Score = 0.3 * Accuracy + 0.7 * mIoU
    """
    return (0.3 * accuracy) + (0.7 * miou)

def print_report(accuracy, miou, final_score=None, model_name="Model"):
    """Wyświetla sformatowany raport."""
    print(f"\nRAPORT DLA: {model_name}")
    print(f"Pixel Accuracy: {accuracy:.2f}%")
    print(f"mIoU:           {miou:.2f}%")
    if final_score is not None:
        print(f"FINAL SCORE:    {final_score:.2f} / 100")
    print("-" * 30)