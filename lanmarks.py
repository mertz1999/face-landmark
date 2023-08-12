import numpy as np
import mediapipe as mp

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


class Landmarks():
    def __init__(self, width, height):
        # instance of face mesh
        mp_facemesh = mp.solutions.face_mesh
        mp_drawing  = mp.solutions.drawing_utils
        self.denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates
        self.face_mesh = mp_facemesh.FaceMesh(refine_landmarks=True)

        self.parts = {
            'lip' : [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
        }

        self.frame_width = width
        self.frame_height = height


    def get(self,frame, part='lip'):
        # Pre-processing
        frame = frame[:, :, ::-1]
        frame = np.ascontiguousarray(frame)

        # Get all landmarks
        results = self.face_mesh.face_meshe.face_mesh.process(frame).multi_face_landmarks
        points = []
        try:
            if results:
                for _, face_landmarks in enumerate(results):
                    marks = face_landmarks.landmark
                    coords_points = []
                    for i in self.parts[part]:
                        lm = marks[i]
                        coord = self.denormalize_coordinates(lm.x, lm.y, self.frame_width, self.frame_height)
                        coords_points.append(coord)
                    
                    points.append(coords_points)

        except:
            return points

        return points


