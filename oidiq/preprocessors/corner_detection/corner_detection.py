from typing import List, Optional, Tuple
from ...session import OIDIQSession, OIDIQPreprocessor, PreProcessors, OIDIQBaseBatchSession
from ...utils import creates, config, OIDIQConfig, get_device, batching, resize_keep_ratio, calculate_4_point_polygon_area
import numpy as np
import cv2
import torch
from .hourglas import PoseNet


class IDCardCornerDetection(OIDIQPreprocessor):
    @config("id_card_corner_detection")
    def init_config(self, config):
        device = get_device(config)
        model_path = config["model_path"]
        model = PoseNet(4, 256, 4)
        # model = torch.load(model_path, map_location=device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.to(device)
        model.eval()
        config["model"] = model
        config["device_obj"] = device

    @creates(PreProcessors.ID_CARD_CORNERS)
    @config("id_card_corner_detection")
    @batching(16)
    def get_corners(self, session: OIDIQBaseBatchSession, config: OIDIQConfig) -> List[np.ndarray]:
        return _run_net(session, config)




def get_net_input(img: np.ndarray, config: OIDIQConfig) -> Tuple[np.ndarray, int, int]:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    size = config.get("input_size", 256)
    pad = config.get("padding", 0.1)
    return resize_keep_ratio(img, (size, size), pad)
    

def get_net_inputs(raw_images: List[np.ndarray], config: OIDIQConfig) -> Tuple[torch.Tensor, List[int], List[int]]:
    imgs = []
    pad_ws = []
    pad_hs = []
    
    for i in range(len(raw_images)):
        img, pad_w, pad_h = get_net_input(raw_images[i], config)
        imgs.append(img)
        pad_ws.append(pad_w)
        pad_hs.append(pad_h)

    imgs = np.array(imgs)
    imgs = imgs[:, np.newaxis, :, :] / 255.0
    imgs *= 2.0
    imgs -= 1.0
    return torch.from_numpy(imgs).float(), pad_ws, pad_hs


def get_corner_positions_from_heatmaps(
    heatmaps: np.ndarray, pad_w: int, pad_h: int, original_size: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], List[float]]:
    positions = []
    confidences = []
    orig_h, orig_w = original_size
    for i in range(4):
        heatmap = heatmaps[0, i, :, :]
        max_val = np.max(heatmap)
        confidences.append(float(max_val))
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Rescale to original corner size
        x -= pad_w
        y -= pad_h
        x = int(float(x) / (heatmap.shape[1] - 2 * pad_w) * orig_w)
        y = int(float(y) / (heatmap.shape[0] - 2 * pad_h) * orig_h)

        positions.append((x, y))

    return positions, confidences


def _detect_corners(raw_images: List[np.ndarray], session: OIDIQBaseBatchSession, config: OIDIQConfig) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    device = config["device_obj"]
    model = config["model"]
    inputs, pad_ws, pad_hs = get_net_inputs(raw_images, config)

    inputs = inputs.to(device)
    with torch.no_grad():
        heatmaps = model(inputs).cpu().numpy()

    results = []
    all_confidences = []
    for i in range(len(raw_images)):

        h,w = raw_images[i].shape[0:2]
        positions, confidences = get_corner_positions_from_heatmaps(heatmaps, pad_ws[i], pad_hs[i], (h, w))

        positions = np.array(positions)
        confidences = np.array(confidences)
        all_confidences.append(confidences)
        thresh = config.get("detection_confidence_threshold", 0.5)
        under_count = np.sum(confidences < thresh)
        session.log(IDCardCornerDetection, f"Detected {4 - under_count}/4 corners with confidences {', '.join([f'{c:.3f}' for c in confidences])} in batch index {i}.")
        positions = estimate_unknown_corners(positions, confidences, thresh, w, h)
        results.append(positions)
    return results, all_confidences, heatmaps

def _run_net(session: OIDIQBaseBatchSession, config: OIDIQConfig) -> List[np.ndarray]:
    raw_images = session.get_raw_image()
    corners, confidences, heatmaps = _detect_corners(raw_images, session, config)

    min_area = config.get("min_area", 0.0)
    if min_area > 0.0:
        to_redetect = []
        bboxes = []
        org_idxs = []
        for i in range(len(corners)):
            h,w = raw_images[i].shape[0:2]
            area = calculate_4_point_polygon_area(corners[i]) / (max(h,w) ** 2)
            if area < min_area and area >= 1:
                session.log(IDCardCornerDetection, f"Redetecting corners in batch index {i} due to small area {area:.3f} < {min_area:.3f}.")
                min_x, max_x = np.min(corners[i][:,0]), np.max(corners[i][:,0])
                min_y, max_y = np.min(corners[i][:,1]), np.max(corners[i][:,1])
                bb_w, bb_h = max_x - min_x, max_y - min_y
                pad = config.get("redetection_padding", 0.05)
                pad_w, pad_h = int(pad * bb_w), int(pad * bb_h)
                start_x = max(0, min_x - pad_w)
                end_x = min(w, max_x + pad_w)
                start_y = max(0, min_y - pad_h)
                end_y = min(h, max_y + pad_h)
                bboxes.append((start_x, start_y, end_x, end_y))
                to_redetect.append(raw_images[i][start_y:end_y, start_x:end_x])
                org_idxs.append(i)
                
        if len(to_redetect) > 0:
            redetected_corners, redetected_confidences, redetected_heatmaps = _detect_corners(to_redetect, session, config)
            for new_corner, new_heatmaps, idx, bbox in zip(redetected_corners, redetected_heatmaps, org_idxs, bboxes):

                start_x, start_y, _, _ = bbox
                new_corner[:,0] += start_x
                new_corner[:,1] += start_y
                corners[idx] = new_corner
                heatmaps[idx] = new_heatmaps


    return corners 
      

def estimate_unknown_corners(corners: np.ndarray, confidences : np.ndarray, thresh:float, w:int, h:int) -> np.ndarray:
    if np.all(confidences >= thresh):
        return corners
    unknown = np.where(confidences < thresh)[0]
    if unknown.size == 1:
        corners[unknown[0]] = estimate_single_missing_corner(corners, unknown[0])
    elif unknown.size == 2:
        if unknown[0] % 2 == unknown[1] % 2:
            corners[unknown[0]] = estimate_single_missing_corner(corners, unknown[0])
            corners[unknown[1]] = estimate_single_missing_corner(corners, unknown[1])
        else:
            corners[unknown[0]], corners[unknown[1]] = estimate_adjacent_missing_corners(corners, unknown[0], unknown[1], w, h)
    elif unknown.size == 3:
        known_index = list(set([0,1,2,3]) - set(unknown))[0]
        if known_index == 0:
            corners[2] = (w-1, h-1)
        elif known_index == 1:
            corners[3] = (0, h-1)
        elif known_index == 2:
            corners[0] = (0, 0)
        else: # known_index ==3
            corners[1] = (w-1, 0)
        corners[(known_index + 1) % 4] = estimate_single_missing_corner(corners, (known_index + 1) % 4)
        corners[(known_index + 3) % 4] = estimate_single_missing_corner(corners, (known_index + 3) % 4)
    else:
        corners[0] = (0, 0)
        corners[1] = (w-1, 0)
        corners[2] = (w-1, h-1)
        corners[3] = (0, h-1)
    return corners
        
        

def estimate_single_missing_corner(corners: np.ndarray, missing_index: int) -> Tuple[int, int]:
    if missing_index % 2 == 0:
        return (corners[(missing_index + 3) % 4][0], corners[(missing_index + 1) % 4][1])
    else:
        return (corners[(missing_index + 1) % 4][0], corners[(missing_index + 3) % 4][1])
    

def estimate_adjacent_missing_corners(corners: np.ndarray, idx1: int, idx2: int, w:int,h:int) -> Tuple[Tuple[int,int],Tuple[int,int]]:
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    elif idx1 == 0 and idx2 == 3:
        idx1, idx2 = idx2, idx1

    if idx1 == 0 and idx2 == 1:
        return (corners[3,0], 0), (corners[2,0], 0)
    elif idx1 == 1 and idx2 == 2:
        return (w-1, corners[0,1]), (w-1, corners[3,1]) 
    elif idx1 == 2 and idx2 == 3:
        return (corners[1,0], h-1), (corners[0,0], h-1)
    else: # idx1 ==3 and idx2 ==0
        return (0, corners[1,1]), (0, corners[2,1])
