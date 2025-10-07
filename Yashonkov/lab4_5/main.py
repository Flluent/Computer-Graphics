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
selected_edges = []  # for highlighting selected edges
window_size = (800, 600)  # initial window size
window_name = "Polygon Geometry Tool"
point_in_poly_results = []  # результаты принадлежности точки полигонам
point_test_mode = False  # режим проверки принадлежности точки полигонам

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
    """Determine position of point B relative to vector OA"""
    xa = A[0] - O[0]
    ya = A[1] - O[1]
    xb = B[0] - O[0]
    yb = B[1] - O[1]
    cross_product = yb * xa - xb * ya
    if cross_product > 0:
        return "B is to the RIGHT of OA"
    elif cross_product < 0:
        return "B is to the LEFT of OA"
    else:
        return "B is ON the line OA"

def is_point_in_polygon(point, polygon):
    """Check if point is inside polygon (any shape, convex or concave)"""
    x, y = point
    n = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-10) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside

# --- Mouse Handler ---
def mouse_callback(event, x, y, flags, param):
    global current_poly, polygons, edges, edge_points, test_point, classification
    global intersection_point, selected_edges, point_in_poly_results, point_test_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # Ctrl + Left Click → classification mode or polygon test point
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            if point_test_mode:
                test_point = (x, y)
                # Проверка принадлежности точки всем полигонам
                point_in_poly_results.clear()
                for poly in polygons:
                    inside = is_point_in_polygon(test_point, poly)
                    point_in_poly_results.append(inside)
                    print(f"Point B {'inside' if inside else 'outside'} polygon {poly.tolist()}")
                point_test_mode = False  # после выбора точки выключаем режим
            else:
                if len(edge_points) < 2:
                    edge_points.append((x, y))
                    print(f"Edge point {len(edge_points)} set: ({x}, {y})")
                elif len(edge_points) == 2 and test_point is None:
                    test_point = (x, y)
                    print(f"Test point B set: ({x}, {y})")
                    O, A = edge_points[0], edge_points[1]
                    classification = classify_point_relative_to_edge(O, A, test_point)
                    print(f"Classification: {classification}")
        else:
            # Normal polygon drawing
            current_poly.append((x, y))
            print(f"Vertex added: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_poly) >= 3:
            polygons.append(np.array(current_poly, np.int32))
            print(f"Polygon completed with {len(current_poly)} vertices")
            current_poly = []
        else:
            print(f"Cannot finish polygon: need at least 3 vertices (currently {len(current_poly)})")

    elif event == cv2.EVENT_MBUTTONDOWN:
        if polygons:
            # Edge selection logic
            min_dist = float('inf')
            nearest_edge = None
            nearest_poly_idx = -1
            nearest_edge_idx = -1
            for poly_idx, poly in enumerate(polygons):
                for i in range(len(poly)):
                    p1 = tuple(poly[i])
                    p2 = tuple(poly[(i + 1) % len(poly)])
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
                        nearest_poly_idx = poly_idx
                        nearest_edge_idx = i
            if nearest_edge and min_dist < 10:
                edges.append(nearest_edge)
                selected_edges.append((nearest_poly_idx, nearest_edge_idx))
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
                    edges = []
                    selected_edges = []

# --- Clear Scene ---
def clear_scene():
    global polygons, current_poly, edges, edge_points, test_point, classification, intersection_point, selected_edges, point_in_poly_results
    polygons = []
    current_poly = []
    edges = []
    edge_points = []
    test_point = None
    classification = None
    intersection_point = None
    selected_edges = []
    point_in_poly_results = []
    print("Scene cleared.")

# --- Reset Classification ---
def reset_classification():
    global edge_points, test_point, classification
    edge_points = []
    test_point = None
    classification = None
    print("Classification reset.")

# --- Resize Window ---
def resize_window(new_width, new_height):
    global window_size
    window_size = (new_width, new_height)
    cv2.resizeWindow(window_name, new_width, new_height)

# --- Main Loop ---
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_size[0], window_size[1])
cv2.setMouseCallback(window_name, mouse_callback)

print("""
Controls:
  Left Click  – Add polygon vertex
  Right Click – Finish current polygon (requires at least 3 vertices)
  Middle Click – Select two edges to check intersection
  Ctrl + Left Click ×2 – Set edge points O and A for classification
  Ctrl + Left Click ×3 – Set test point B and get classification
  P – Activate point-in-polygon test mode
  C – Clear scene
  R – Reset classification only
  + / = – Increase window size
  - – Decrease window size
  Q – Quit
""")

while True:
    img = np.ones((window_size[1], window_size[0], 3), np.uint8) * 255

    # Draw polygons, edges, vertices
    for poly_idx, poly in enumerate(polygons):
        if len(poly) == 1:
            cv2.circle(img, tuple(poly[0]), 3, (0, 0, 255), -1)
        elif len(poly) == 2:
            cv2.line(img, tuple(poly[0]), tuple(poly[1]), (0, 0, 255), 2)
        else:
            cv2.polylines(img, [poly], True, (0, 0, 255), 2)

    for poly_idx, edge_idx in selected_edges:
        if poly_idx < len(polygons):
            poly = polygons[poly_idx]
            if len(poly) > 1:
                p1 = tuple(poly[edge_idx])
                p2 = tuple(poly[(edge_idx + 1) % len(poly)])
                cv2.line(img, p1, p2, (255, 0, 255), 4)
                cv2.circle(img, p1, 6, (255, 0, 255), -1)
                cv2.circle(img, p2, 6, (255, 0, 255), -1)

    if len(current_poly) > 1:
        cv2.polylines(img, [np.array(current_poly)], False, (0, 255, 0), 1)
    elif len(current_poly) == 1:
        cv2.circle(img, current_poly[0], 3, (0, 255, 0), -1)

    # Draw classification points
    if len(edge_points) >= 1:
        cv2.circle(img, edge_points[0], 6, (255, 0, 0), -1)
        cv2.putText(img, "O", (edge_points[0][0] + 5, edge_points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
    if len(edge_points) == 2:
        cv2.circle(img, edge_points[1], 6, (0, 100, 255), -1)
        cv2.putText(img, "A", (edge_points[1][0] + 5, edge_points[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,100,255),1)
        cv2.arrowedLine(img, edge_points[0], edge_points[1], (0,100,255), 2)

    if test_point is not None:
        color = (0, 255, 0)
        if classification and "LEFT" in classification:
            color = (255,0,0)
        elif classification and "RIGHT" in classification:
            color = (0,0,255)
        elif classification and "ON" in classification:
            color = (0,150,0)
        cv2.circle(img, test_point, 6, color, -1)
        cv2.putText(img, "B", (test_point[0]+5, test_point[1]), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        if classification:
            cv2.putText(img, classification, (50,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

    # Draw intersection
    if intersection_point:
        cv2.circle(img, intersection_point, 6, (0, 255, 0), -1)
        cv2.putText(img, "Intersection", (intersection_point[0]+10, intersection_point[1]), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,150,0),2)

    # Show current vertex info
    if len(current_poly) > 0:
        vertex_info = f"Current vertices: {len(current_poly)} (need {3 - len(current_poly)} more to finish)"
        cv2.putText(img, vertex_info, (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    # Show selected edges info
    if selected_edges:
        selection_info = f"Selected edges: {len(selected_edges)}/2"
        cv2.putText(img, selection_info, (10,55), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

    # Show point-in-polygon results
    if test_point is not None and point_in_poly_results:
        for idx, inside in enumerate(point_in_poly_results):
            color = (0,150,0) if inside else (0,0,150)
            text = f"B {'inside' if inside else 'outside'} poly {idx}"
            cv2.putText(img, text, (10, 80 + idx*20), cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

    # Window size info
    size_info = f"Window: {window_size[0]}x{window_size[1]} (+/- to resize)"
    cv2.putText(img, size_info, (10, window_size[1]-20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,100),1)

    # Instructions
    instructions = [
        "Ctrl+Click: Set O, A, B for classification",
        "Middle Click: Select edges for intersection (magenta)",
        "P: Activate point-in-polygon test",
        "C: Clear, R: Reset classification, Q: Quit",
        "+/-: Resize window"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, window_size[1]-100+i*20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    cv2.imshow(window_name, img)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        clear_scene()
    elif key == ord('r'):
        reset_classification()
    elif key == ord('p'):
        point_test_mode = True
        print("Point-in-polygon test mode activated. Click Ctrl+Left to set point B.")
    elif key in [ord('+'), ord('=')]:
        new_width = min(2000, window_size[0]+100)
        new_height = min(1500, window_size[1]+100)
        resize_window(new_width, new_height)
        print(f"Window resized to: {new_width}x{new_height}")
    elif key == ord('-'):
        new_width = max(400, window_size[0]-100)
        new_height = max(300, window_size[1]-100)
        resize_window(new_width, new_height)
        print(f"Window resized to: {new_width}x{new_height}")

cv2.destroyAllWindows()
