import cv2
import numpy as np

# --- Global Variables ---
polygons = []  # completed polygons
current_poly = []  # polygon under construction
edges = []  # selected edges for intersection
edge_points = []  # two points for classification: O and A
test_point = None  # test point B for classification
classification = None
intersection_point = None  # for visualization

window_name = "Polygon Geometry Tool"


# --- Math Functions ---
def segment_intersection(p1, p2, p3, p4):
    """Return intersection point of two segments, or None if they don't intersect"""

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (int(x), int(y))


def classify_point_relative_to_edge(O, A, B):
    """
    Determine the position of point B relative to vector OA
    In computer graphics (Y axis down), the signs are reversed:
    - Positive value = point is to the RIGHT
    - Negative value = point is to the LEFT
    """
    xa = A[0] - O[0]  # vector OA x-component
    ya = A[1] - O[1]  # vector OA y-component
    xb = B[0] - O[0]  # vector OB x-component
    yb = B[1] - O[1]  # vector OB y-component

    cross_product = yb * xa - xb * ya  # векторное произведение

    # В компьютерной графике (ось Y вниз) знаки обратные
    if cross_product > 0:
        return "B is to the RIGHT of OA"  # было LEFT
    elif cross_product < 0:
        return "B is to the LEFT of OA"  # было RIGHT
    else:
        return "B is ON the line OA"


# --- Mouse Handler ---
def mouse_callback(event, x, y, flags, param):
    global current_poly, polygons, edges, edge_points, test_point, classification, intersection_point

    if event == cv2.EVENT_LBUTTONDOWN:
        # Ctrl + Left Click → classification mode
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if len(edge_points) < 2:
                edge_points.append((x, y))
                print(f"Edge point {len(edge_points)} set: ({x}, {y})")
            elif len(edge_points) == 2 and test_point is None:
                test_point = (x, y)
                print(f"Test point B set: ({x}, {y})")
                # Perform classification
                if len(edge_points) == 2:
                    O, A = edge_points[0], edge_points[1]
                    classification = classify_point_relative_to_edge(O, A, test_point)
                    print(f"Classification: {classification}")
                    print(f"O={O}, A={A}, B={test_point}")
        else:
            # Normal polygon drawing
            current_poly.append((x, y))
            print(f"Vertex added: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Finish polygon
        if len(current_poly) >= 3:
            polygons.append(np.array(current_poly, np.int32))
            print(f"Polygon completed with {len(current_poly)} vertices")
        elif len(current_poly) == 2:
            polygons.append(np.array(current_poly, np.int32))
            print("Edge created")
        elif len(current_poly) == 1:
            polygons.append(np.array(current_poly, np.int32))
            print("Point created")
        current_poly = []

    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle click → select edges for intersection
        if polygons:
            min_dist = float('inf')
            nearest_edge = None
            for poly in polygons:
                for i in range(len(poly)):
                    p1 = tuple(poly[i])
                    p2 = tuple(poly[(i + 1) % len(poly)])
                    # Calculate distance to edge
                    edge_vec = (p2[0] - p1[0], p2[1] - p1[1])
                    to_point = (x - p1[0], y - p1[1])
                    edge_length_sq = edge_vec[0] ** 2 + edge_vec[1] ** 2

                    if edge_length_sq == 0:
                        continue

                    t = max(0, min(1, (to_point[0] * edge_vec[0] + to_point[1] * edge_vec[1]) / edge_length_sq))
                    projection = (p1[0] + t * edge_vec[0], p1[1] + t * edge_vec[1])
                    dist = np.sqrt((x - projection[0]) ** 2 + (y - projection[1]) ** 2)

                    if dist < min_dist:
                        min_dist = dist
                        nearest_edge = (p1, p2)

            if nearest_edge and min_dist < 10:  # threshold for selection
                edges.append(nearest_edge)
                print(f"Edge selected: {nearest_edge}")
                if len(edges) == 2:
                    p1, p2 = edges[0]
                    p3, p4 = edges[1]
                    intersection_point = segment_intersection(p1, p2, p3, p4)
                    if intersection_point:
                        print(f"Intersection found at: {intersection_point}")
                    else:
                        print("Edges do not intersect.")
                        intersection_point = None
                    edges = []  # reset for next selection


# --- Clear Scene ---
def clear_scene():
    global polygons, current_poly, edges, edge_points, test_point, classification, intersection_point
    polygons = []
    current_poly = []
    edges = []
    edge_points = []
    test_point = None
    classification = None
    intersection_point = None
    print("Scene cleared.")


# --- Reset Classification ---
def reset_classification():
    global edge_points, test_point, classification
    edge_points = []
    test_point = None
    classification = None
    print("Classification reset.")


# --- Main Loop ---
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("""
Controls:
  Left Click  – Add polygon vertex
  Right Click – Finish current polygon
  Middle Click – Select two edges to check intersection
  Ctrl + Left Click ×2 – Set edge points O and A for classification
  Ctrl + Left Click ×3 – Set test point B and get classification
  C – Clear scene
  R – Reset classification only
  Q – Quit
""")

while True:
    img = np.ones((600, 800, 3), np.uint8) * 255

    # Draw all polygons
    for poly in polygons:
        if len(poly) == 1:
            cv2.circle(img, tuple(poly[0]), 3, (0, 0, 255), -1)
        elif len(poly) == 2:
            cv2.line(img, tuple(poly[0]), tuple(poly[1]), (0, 0, 255), 2)
        else:
            cv2.polylines(img, [poly], True, (0, 0, 255), 2)

    # Draw current polygon
    if len(current_poly) > 1:
        cv2.polylines(img, [np.array(current_poly)], False, (0, 255, 0), 1)
    elif len(current_poly) == 1:
        cv2.circle(img, current_poly[0], 3, (0, 255, 0), -1)

    # Draw classification edge and test point
    if len(edge_points) >= 1:
        cv2.circle(img, edge_points[0], 6, (255, 0, 0), -1)
        cv2.putText(img, "O", (edge_points[0][0] + 5, edge_points[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if len(edge_points) == 2:
        cv2.circle(img, edge_points[1], 6, (0, 100, 255), -1)
        cv2.putText(img, "A", (edge_points[1][0] + 5, edge_points[1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        cv2.arrowedLine(img, edge_points[0], edge_points[1], (0, 100, 255), 2)

    if test_point is not None:
        color = (0, 255, 0)
        if classification and "LEFT" in classification:
            color = (255, 0, 0)  # Red for left
        elif classification and "RIGHT" in classification:
            color = (0, 0, 255)  # Blue for right
        elif classification and "ON" in classification:
            color = (0, 150, 0)  # Green for on line

        cv2.circle(img, test_point, 6, color, -1)
        cv2.putText(img, "B", (test_point[0] + 5, test_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if classification:
            cv2.putText(img, classification, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw intersection point
    if intersection_point:
        cv2.circle(img, intersection_point, 6, (0, 255, 0), -1)
        cv2.putText(img, "Intersection", (intersection_point[0] + 10, intersection_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    # Draw instructions
    instructions = [
        "Ctrl+Click: Set O, A, B for classification",
        "Middle Click: Select edges for intersection",
        "C: Clear, R: Reset classification, Q: Quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, 550 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_scene()
    elif key == ord('r'):
        reset_classification()

cv2.destroyAllWindows()