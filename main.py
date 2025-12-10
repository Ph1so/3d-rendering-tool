# Requirements:
# PyOpenGL 
# PyOpenGL_accelerate 
# glfw 
# numpy
# imgui
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import math



# GEOMETRY DEFINITIONS
def make_cube():
    vertices = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ], dtype=np.float32)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    return vertices, edges


def make_pyramid():
    vertices = np.array([
        [-1, -1, -1],  # base
        [ 1, -1, -1],
        [ 1, -1,  1],
        [-1, -1,  1],
        [ 0,  1,  0],  # tip
    ], dtype=np.float32)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (0,4),(1,4),(2,4),(3,4)
    ]
    return vertices, edges


def make_cylinder(n=20):
    verts = []
    # bottom circle
    for i in range(n):
        ang = 2*np.pi*i/n
        verts.append([math.cos(ang), -1, math.sin(ang)])

    # top circle
    for i in range(n):
        ang = 2*np.pi*i/n
        verts.append([math.cos(ang),  1, math.sin(ang)])

    vertices = np.array(verts, dtype=np.float32)
    edges = []

    # bottom ring
    for i in range(n):
        edges.append((i, (i+1) % n))
    # top ring
    for i in range(n):
        edges.append((i+n, ((i+1) % n) + n))
    # verticals
    for i in range(n):
        edges.append((i, i+n))

    return vertices, edges



# DRAW HELPERS
def draw_edges(vertices, edges):
    glBegin(GL_LINES)
    glColor3f(1, 1, 1)
    for a, b in edges:
        glVertex3fv(vertices[a])
        glVertex3fv(vertices[b])
    glEnd()


def draw_selected_vertex(vertices, index):
    if 0 <= index < len(vertices):
        glPointSize(14)
        glBegin(GL_POINTS)
        glColor3f(1.0, 1.0, 0.0)  # yellow
        glVertex3fv(vertices[index])
        glEnd()


def draw_first_edge_vertex(vertices, index):
    if 0 <= index < len(vertices):
        glPointSize(14)
        glBegin(GL_POINTS)
        glColor3f(0.0, 1.0, 0.0)  # green
        glVertex3fv(vertices[index])
        glEnd()


def draw_highlighted_edge(vertices, edges, edge_index):
    if 0 <= edge_index < len(edges):
        a, b = edges[edge_index]
        glLineWidth(4)
        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 1.0)  # cyan
        glVertex3fv(vertices[a])
        glVertex3fv(vertices[b])
        glEnd()
        glLineWidth(1)

def draw_world_axes(length=2.0):
    glLineWidth(3)
    glBegin(GL_LINES)

    # X axis (red)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    # Y axis (green)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    # Z axis (blue)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, length)

    glEnd()
    glLineWidth(1)

def draw_text(x, y, text, size=GLUT_BITMAP_9_BY_15):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 800, 0, 600)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glColor3f(1, 1, 1)
    glRasterPos2f(x, y)

    for ch in text:
        glutBitmapCharacter(size, ord(ch))

    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# MAIN
def main():
    if not glfw.init():
        raise Exception("Could not init GLFW")

    window = glfw.create_window(800, 600, "3D Tool + SVD Visualizer", None, None)
    glfw.make_context_current(window)
    glutInit()

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 800/600, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)

    
    # CAMERA
    cam_pos  = np.array([0.0,0.0,5.0],dtype=np.float32)
    cam_yaw  = 0.0
    cam_pitch = 0.0

    
    # OBJECTS
    cube     = make_cube()
    pyramid  = make_pyramid()
    cylinder = make_cylinder(20)

    objects = [cube,pyramid,cylinder]
    object_names = ["Cube","Pyramid","Cylinder"]
    current_object = 0

    vertices, edges = objects[current_object]

    
    # STATE FLAGS
    rotate_mode   = False
    scale_mode    = False
    vertex_mode   = False
    edge_mode     = False
    edge_create_mode = False
    show_instructions = False

    r_press = e_press = v_press = q_press = j_press = False
    m_press = False
    c_press = False

    selected_vertex = 0
    selected_edge   = 0
    first_vertex_selected = None

    tab_locked   = False
    enter_locked = False

    cube_rot_x = 0
    cube_rot_y = 0
    cube_scale = 1.0

    
    # SVD VISUALIZER STATE
    svd_M = np.array([
        [1.2, 0.4, 0.1],
        [0.0, 0.9, 0.3],
        [0.2, 0.0, 1.1]
    ], dtype=np.float32)

    svd_U, svd_S, svd_Vt = np.linalg.svd(svd_M)

    svd_mode = False
    svd_step = 0
    svd_animating = False
    svd_anim_phase = 0
    svd_anim_t = 0.0

    t_press = False
    key1_press = key2_press = key3_press = key4_press = False

    
    while not glfw.window_should_close(window):
        # M = Instructions Page
        if glfw.get_key(window, glfw.KEY_M) == glfw.PRESS:
            if not m_press:
                show_instructions = not show_instructions
            m_press = True
        else:
            m_press = False

        if show_instructions:
            glClearColor(0,0,0,1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            y = 550
            dy = -25
            def line(txt):
                nonlocal y
                draw_text(50, y, txt)
                y += dy

            draw_text(50,580,"=== CONTROLS ===", GLUT_BITMAP_HELVETICA_18)
            line("M : Toggle this help page")
            line("W/A/S/D : Move Camera")
            line("Arrow Keys : Rotate Camera")
            line("R : Rotate Mode")
            line("E : Scale Mode")
            line("V : Vertex Edit Mode")
            line("Q : Edge Select Mode")
            line("C : Edge Create Mode")
            line("TAB / Shift+TAB : Cycle vertices/edges")
            line("ENTER : Insert or create edges")
            line("J : Switch objects")
            line("")
            line("T : Toggle SVD Mode")
            line("  1 : Show V^T")
            line("  2 : Show ΣV^T")
            line("  3 : Show UΣV^T")
            line("  4 : Animate (now slower)")
            line("")
            draw_text(50, 80, "Press M to return.")

            glfw.swap_buffers(window)
            glfw.poll_events()
            continue

       
        # Normal Rendering
        glClearColor(0,0,0,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        move_speed   = 0.025
        rot_speed    = 0.25
        vertex_speed = 0.05

        
        # OBJECT SWITCHING (J)
        if glfw.get_key(window,glfw.KEY_J)==glfw.PRESS:
            if not j_press:
                current_object = (current_object+1)%len(objects)
                vertices,edges = objects[current_object]
                selected_vertex = 0
                selected_edge   = 0
                first_vertex_selected = None
            j_press=True
        else:
            j_press=False

        
        # Mode toggles (Disable SVD)
        if glfw.get_key(window,glfw.KEY_R)==glfw.PRESS:
            if not r_press:
                rotate_mode = not rotate_mode
                scale_mode = False
                vertex_mode = False
                edge_mode = False
                edge_create_mode = False
                svd_mode = False
            r_press=True
        else:
            r_press=False

        if glfw.get_key(window,glfw.KEY_E)==glfw.PRESS:
            if not e_press:
                scale_mode = not scale_mode
                rotate_mode = False
                vertex_mode = False
                edge_mode = False
                edge_create_mode = False
                svd_mode = False
            e_press=True
        else:
            e_press=False

        if glfw.get_key(window,glfw.KEY_V)==glfw.PRESS:
            if not v_press:
                vertex_mode = not vertex_mode
                edge_mode = False
                rotate_mode = False
                scale_mode = False
                edge_create_mode = False
                svd_mode = False
            v_press=True
        else:
            v_press=False

        if glfw.get_key(window,glfw.KEY_Q)==glfw.PRESS:
            if not q_press:
                edge_mode = not edge_mode
                vertex_mode = False
                rotate_mode = False
                scale_mode = False
                edge_create_mode = False
                svd_mode = False
            q_press=True
        else:
            q_press=False

        if glfw.get_key(window,glfw.KEY_C)==glfw.PRESS:
            if not c_press:
                edge_create_mode = not edge_create_mode
                first_vertex_selected = None
                vertex_mode = False
                edge_mode = False
                rotate_mode = False
                scale_mode = False
                svd_mode = False
            c_press=True
        else:
            c_press=False

        
        # SVD MODE TOGGLE (T)
        if glfw.get_key(window, glfw.KEY_T) == glfw.PRESS:
            if not t_press:
                svd_mode = not svd_mode
                svd_animating = False
                svd_step = 0
                svd_anim_phase = 0
                svd_anim_t = 0.0

                # Disable editing modes, but DO NOT disable camera rotation
                if svd_mode:
                    rotate_mode = False
                    scale_mode = False
                    vertex_mode = False
                    edge_mode = False
                    edge_create_mode = False

            t_press = True
        else:
            t_press = False

        
        # TAB cycling
        tab = glfw.get_key(window, glfw.KEY_TAB)==glfw.PRESS
        if tab and not tab_locked:
            tab_locked=True
            shift = (glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or 
                     glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS)

            if edge_mode and len(edges)>0:
                selected_edge = (selected_edge - 1) % len(edges) if shift else (selected_edge + 1) % len(edges)
            else:
                if len(vertices) > 0:
                    selected_vertex = (selected_vertex - 1) % len(vertices) if shift else (selected_vertex + 1) % len(vertices)

        if not tab:
            tab_locked=False

        
        # ENTER
        enter_pressed = (
            glfw.get_key(window,glfw.KEY_ENTER)==glfw.PRESS or 
            glfw.get_key(window,glfw.KEY_KP_ENTER)==glfw.PRESS
        )

        if edge_mode and enter_pressed and not enter_locked and len(edges)>0:
            enter_locked=True
            eA,eB = edges[selected_edge]
            vA, vB = vertices[eA], vertices[eB]

            new_v = (vA+vB)/2.0
            vertices = np.vstack([vertices,new_v])
            new_idx = len(vertices)-1

            edges.pop(selected_edge)
            edges.insert(selected_edge,(eA,new_idx))
            edges.insert(selected_edge+1,(new_idx,eB))
            objects[current_object] = (vertices,edges)

            vertex_mode=True
            edge_mode=False
            edge_create_mode=False
            svd_mode = False
            selected_vertex=new_idx

        if edge_create_mode and enter_pressed and not enter_locked:
            enter_locked = True
            if first_vertex_selected is None:
                first_vertex_selected = selected_vertex
            else:
                vA = first_vertex_selected
                vB = selected_vertex
                if vA != vB and (vA,vB) not in edges and (vB,vA) not in edges:
                    edges.append((vA,vB))
                edge_create_mode = False
                first_vertex_selected = None

        if not enter_pressed:
            enter_locked=False

        
        # CAMERA ORIENTATION
        yaw=np.radians(cam_yaw)
        pitch=np.radians(cam_pitch)

        forward=np.array([
            np.sin(yaw)*np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw)*np.cos(pitch)
        ],dtype=np.float32)
        forward/=np.linalg.norm(forward)

        world_up=np.array([0,1,0],dtype=np.float32)
        right=np.cross(forward,world_up); right/=np.linalg.norm(right)
        up=np.cross(right,forward); up/=np.linalg.norm(up)

        
        # APPLY CAMERA
        glLoadIdentity()
        center=cam_pos+forward
        gluLookAt(
            cam_pos[0],cam_pos[1],cam_pos[2],
            center[0],center[1],center[2],
            up[0],up[1],up[2]
        )

        
        # CAMERA MOVEMENT (WASD)
        keys=[]
        if glfw.get_key(window,glfw.KEY_W)==glfw.PRESS:
            cam_pos+=forward*move_speed; keys.append("W")
        if glfw.get_key(window,glfw.KEY_S)==glfw.PRESS:
            cam_pos-=forward*move_speed; keys.append("S")
        if glfw.get_key(window,glfw.KEY_A)==glfw.PRESS:
            cam_pos-=right*move_speed;   keys.append("A")
        if glfw.get_key(window,glfw.KEY_D)==glfw.PRESS:
            cam_pos+=right*move_speed;   keys.append("D")

        
        # CAMERA ROTATION (ALWAYS enabled, except editing modes)
        editing_block = (rotate_mode or scale_mode or vertex_mode or edge_mode or edge_create_mode)

        if not editing_block:
            if glfw.get_key(window,glfw.KEY_LEFT)==glfw.PRESS:
                cam_yaw-=rot_speed
            if glfw.get_key(window,glfw.KEY_RIGHT)==glfw.PRESS:
                cam_yaw+=rot_speed
            if glfw.get_key(window,glfw.KEY_UP)==glfw.PRESS:
                cam_pitch+=rot_speed
            if glfw.get_key(window,glfw.KEY_DOWN)==glfw.PRESS:
                cam_pitch-=rot_speed
            cam_pitch=np.clip(cam_pitch,-89,89)

        
        # VERTEX EDIT
        if vertex_mode and len(vertices) > 0:
            if glfw.get_key(window,glfw.KEY_UP)==glfw.PRESS:
                vertices[selected_vertex][1]+=vertex_speed
            if glfw.get_key(window,glfw.KEY_DOWN)==glfw.PRESS:
                vertices[selected_vertex][1]-=vertex_speed
            if glfw.get_key(window,glfw.KEY_LEFT)==glfw.PRESS:
                vertices[selected_vertex][0]-=vertex_speed
            if glfw.get_key(window,glfw.KEY_RIGHT)==glfw.PRESS:
                vertices[selected_vertex][0]+=vertex_speed

        if rotate_mode:
            if glfw.get_key(window,glfw.KEY_LEFT)==glfw.PRESS:
                cube_rot_y -= 2
            if glfw.get_key(window,glfw.KEY_RIGHT)==glfw.PRESS:
                cube_rot_y += 2
            if glfw.get_key(window,glfw.KEY_UP)==glfw.PRESS:
                cube_rot_x -= 2
            if glfw.get_key(window,glfw.KEY_DOWN)==glfw.PRESS:
                cube_rot_x += 2
        
        
        # OBJECT SCALING (E mode)
        if scale_mode:
            if glfw.get_key(window,glfw.KEY_UP)==glfw.PRESS:
                cube_scale *= 1.02
            if glfw.get_key(window,glfw.KEY_DOWN)==glfw.PRESS:
                cube_scale *= 0.98

        objects[current_object]=(vertices,edges)

        
        # SVD STEP KEYS (1,2,3,4)
        if svd_mode:
            if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
                if not key1_press:
                    svd_step = 1
                    svd_animating=False
                key1_press = True
            else: key1_press=False

            if glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
                if not key2_press:
                    svd_step = 2
                    svd_animating=False
                key2_press = True
            else: key2_press=False

            if glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
                if not key3_press:
                    svd_step = 3
                    svd_animating=False
                key3_press = True
            else: key3_press=False

            if glfw.get_key(window, glfw.KEY_4) == glfw.PRESS:
                if not key4_press:
                    svd_animating = not svd_animating
                    svd_anim_phase = 0
                    svd_anim_t = 0.0
                    svd_step = 0
                key4_press = True
            else: key4_press=False

        
        # BUILD SVD TRANSFORMED VERTICES
        render_vertices = vertices
        centroid = None

        if svd_mode and len(vertices) > 0:
            centroid = vertices.mean(axis=0)
            Xc = vertices - centroid

            I3  = np.eye(3, dtype=np.float32)
            Vt  = svd_Vt
            Sig = np.diag(svd_S)
            U   = svd_U

            A1 = Vt
            A2 = Sig @ Vt
            A3 = U @ Sig @ Vt

            if svd_animating:
                if svd_anim_phase == 0: A_from, A_to = I3, A1
                elif svd_anim_phase == 1: A_from, A_to = A1, A2
                elif svd_anim_phase == 2: A_from, A_to = A2, A3
                else: A_from, A_to = A3, I3

                svd_anim_t += 0.005

                alpha = svd_anim_t
                A_current = (1-alpha)*A_from + alpha*A_to

                if svd_anim_t >= 1.0:
                    svd_anim_t = 0.0
                    svd_anim_phase = (svd_anim_phase+1) % 4
            else:
                if   svd_step == 0: A_current = I3
                elif svd_step == 1: A_current = A1
                elif svd_step == 2: A_current = A2
                elif svd_step == 3: A_current = A3
                else: A_current = I3

            render_vertices = Xc @ A_current.T + centroid

        
        # DRAW SCENE       
        glPushMatrix()

        # In SVD mode skip local rotate/scale
        if not svd_mode:
            glScalef(cube_scale,cube_scale,cube_scale)
            glRotatef(cube_rot_x,1,0,0)
            glRotatef(cube_rot_y,0,1,0)

        draw_edges(render_vertices,edges)
        draw_selected_vertex(render_vertices,selected_vertex)

        if edge_mode:
            draw_highlighted_edge(render_vertices,edges,selected_edge)
        if edge_create_mode and first_vertex_selected is not None:
            draw_first_edge_vertex(render_vertices, first_vertex_selected)

        # Draw singular axes in SVD mode
        if svd_mode and centroid is not None:
            V = svd_Vt.T
            glLineWidth(3)
            glBegin(GL_LINES)
            for i in range(3):
                axis = V[:, i]
                scale = svd_S[i] * 1.5
                if i == 0: glColor3f(1,0,0)
                elif i == 1: glColor3f(0,1,0)
                else: glColor3f(0,0,1)
                start = centroid
                end = centroid + axis * scale
                glVertex3f(*start)
                glVertex3f(*end)
            glEnd()
            glLineWidth(1)

        glDisable(GL_DEPTH_TEST)
        draw_world_axes(2.0)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()

        
        # HUD
        draw_text(10,570,f"Camera: {cam_pos[0]:.2f} {cam_pos[1]:.2f} {cam_pos[2]:.2f}")
        draw_text(10,545,"Keys: "+(", ".join(keys) if keys else "(none)"))
        draw_text(10,520,f"Object: {object_names[current_object]} (J to switch)")
        draw_text(10,495,f"Selected Vertex: {selected_vertex}")

        if edge_mode and len(edges)>0:
            a,b = edges[selected_edge]
            draw_text(10,470,f"Selected Edge: {selected_edge} (v{a} - v{b})")

        if edge_create_mode:
            if first_vertex_selected is None:
                draw_text(10,470,"Edge Create: select FIRST vertex, ENTER to confirm")
            else:
                draw_text(10,470,f"Edge Create: first={first_vertex_selected}, choose SECOND, ENTER")

        # Mode display
        if svd_mode:
            anim_state = "ON" if svd_animating else "OFF"
            mode_txt = f"Mode: SVD (T) | 1:V^T  2:ΣV^T  3:UΣV^T  4:Animate [{anim_state}]"
        elif vertex_mode:
            mode_txt = "Mode: Vertex Edit (V)"
        elif edge_mode:
            mode_txt = "Mode: Edge Select (Q)"
        elif edge_create_mode:
            mode_txt = "Mode: Edge Create (C)"
        elif rotate_mode:
            mode_txt = "Mode: Rotate (R)"
        elif scale_mode:
            mode_txt = "Mode: Scale (E)"
        else:
            mode_txt = "Mode: Camera (WASD + Arrow Keys)"

        draw_text(10,450,mode_txt)
        draw_text(10,425,"Press M for help/instructions")

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
