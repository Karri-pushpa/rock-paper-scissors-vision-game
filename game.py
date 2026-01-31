import cv2
import mediapipe as mp
import random
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_up(hand_landmarks, handedness="Right"):
    lm = hand_landmarks.landmark
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    fingers = []

    # Thumb
    if handedness == "Right":
        fingers.append(lm[4].x < lm[2].x)
    else:
        fingers.append(lm[4].x > lm[2].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(lm[tips_ids[i]].y < lm[pip_ids[i]].y)

    return fingers

def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if all(fingers):
        return "paper"
    if index and middle and not ring and not pinky:
        return "scissors"
    if sum(fingers) <= 1:
        return "rock"
    return "unknown"

def decide_winner(player, comp):
    if player == comp:
        return "Tie", 0
    wins = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    if player in wins and wins[player] == comp:
        return "You Win!", 1
    else:
        return "You Lose", -1

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    player_score, comp_score, rounds = 0, 0, 0
    last_round_time = 0.0
    cooldown = 2.5
    outcome_text = ""

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            player_choice = "No hand"
            # If a hand is detected, process it
            if result.multi_hand_landmarks:
                # use the first detected hand
                hand_landmarks = result.multi_hand_landmarks[0]

                # get handedness if available
                handedness = "Right"
                if result.multi_handedness and len(result.multi_handedness) > 0:
                    handedness = result.multi_handedness[0].classification[0].label

                # classify fingers -> gesture
                fingers = fingers_up(hand_landmarks, handedness)
                gesture = classify_gesture(fingers)

                # Custom hand landmark colors: red points, white connections
                landmark_style = mp_drawing.DrawingSpec(color=(255,0, 0), thickness=3, circle_radius=1)   
                connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)             # White
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style
                )

                now = time.time()
                if now - last_round_time > cooldown and gesture in ("rock", "paper", "scissors"):
                    last_round_time = now
                    player_choice = gesture
                    comp_choice = random.choice(["rock", "paper", "scissors"])
                    result_text, delta = decide_winner(player_choice, comp_choice)
                    if delta == 1:
                        player_score += 1
                    elif delta == -1:
                        comp_score += 1
                    rounds += 1
                    outcome_text = f"{result_text} ({player_choice} vs {comp_choice})"
                else:
                    player_choice = gesture

            # ---- Overlay UI (colored) ----
            # Rounds (Red)
            cv2.putText(frame, f"Rounds: {rounds}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Your Score (Green)
            cv2.putText(frame, f"You: {player_score}", (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # CPU Score (Blue)
            cv2.putText(frame, f"CPU: {comp_score}", (360, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Detected gesture (near bottom)
            cv2.putText(frame, f"Detected: {player_choice}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2)

            # Last outcome
            if outcome_text:
                cv2.putText(frame, outcome_text, (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("RPS Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                player_score = comp_score = rounds = 0
                outcome_text = ""
                last_round_time = 0.0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

