"""Show YOLO26 detections from camera 0; press q in the window to quit."""

from typing import cast

import cv2
import torch

from foveayolo import YOLO26


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = YOLO26.from_pretrained("yolo26n.pt", device=device).eval().fuse()
    model.end2end = True
    camera = cv2.VideoCapture(0)

    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open camera 0")

        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0)
            x = x.to(device=device, dtype=torch.float32) / 255

            with torch.inference_mode():
                detections, _ = cast(tuple[torch.Tensor, object], model(x))

            for row in detections[0].cpu().unbind(0):
                x1, y1, x2, y2, score, cls = row.unbind(0)
                confidence = float(score)
                if confidence < 0.25:
                    continue

                left, top, right, bottom = map(int, (x1, y1, x2, y2))
                _ = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                _ = cv2.putText(
                    frame,
                    f"{int(cls)}: {confidence:.2f}",
                    (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("FoveaYOLO - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
