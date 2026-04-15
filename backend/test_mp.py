import mediapipe as mp
print(dir(mp))          # Should include 'solutions'
print(mp.solutions.face_mesh)   # Should print a module object