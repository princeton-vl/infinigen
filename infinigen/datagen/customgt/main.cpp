// Copyright (c) Princeton University.
// This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

// Authors: Lahav Lipson


#include <glad/glad.h>
#if SYSTEM_NUM == 0
    #include <EGL/egl.h>
    #include <EGL/eglext.h>
#elif SYSTEM_NUM == 1
    #include <GLFW/glfw3.h>
#endif
#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <stack>
#include <chrono>
#include <regex>
#include <math.h>
#include <indicators/progress_bar.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include "shader.hpp"
#include "blender_object.hpp"
#include "camera_view.hpp"
#include "string_tools.hpp"
#include "load_blender_mesh.hpp"
#include "utils.hpp"
#include "io.hpp"

#define VERSION "1.43"

using std::cout, std::cerr, std::endl;

#if SYSTEM_NUM == 0
static const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE,
    EGL_CONFORMANT, EGL_OPENGL_ES2_BIT,
};
#elif SYSTEM_NUM == 1
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
#endif

template <typename T>
std::vector<T> read_buffer(GLenum color_attachment, const int width, const int height){
    std::vector<T> pixels(width * height * 4);
    static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value);
    GLint format = 0, type = 0;
    glReadBuffer(color_attachment);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &type);
    const std::string message = "type=" + std::to_string(type) + " format=" + std::to_string(format);
    if constexpr(std::is_same<T, float>::value)
        MRASSERT((type == GL_FLOAT) && (format == GL_RGBA), message);
    else
        MRASSERT((type == GL_INT) && (format == GL_RGBA_INTEGER), message);
    glReadPixels(0, 0, width, height, format, type, pixels.data());
    return pixels;
}

// From https://stackoverflow.com/a/10791845/5057543
#define XSTR(x) STR(x)
#define STR(x) #x

int main(int argc, char *argv[]) {

    const fs::path source_directory = XSTR(PROJECT_SOURCE_DIR) ;
    const auto cpp_path = source_directory / "main.cpp";
    assert_exists(cpp_path);
    std::ifstream t(cpp_path.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    const auto file_text = buffer.str();
    const std::regex regex{"VERSION \\\"([0-9\\.]+)\\\""};
    std::smatch m;
    if ((std::regex_search(file_text, m, regex)) && (VERSION != m[1])){
        std::cerr << "Error: The customgt executable is out-of-date, you need to re-compile it." << std::endl;
        exit(1);
    }

    setenv("MESA_GL_VERSION_OVERRIDE", "3.3", true);

    argparse::ArgumentParser program("main", VERSION);
    program.add_argument("--frame").required().help("Current frame").scan<'i', int>();
    program.add_argument("-in", "--input_dir").required().help("The input/output dir");
    program.add_argument("-out", "--output_dir").required().help("The input/output dir");
    program.add_argument("-lw", "--line_width").default_value(2).help("The width of the occlusion boundaries").scan<'i', int>();
    program.add_argument("-s", "--subdivide").default_value(2).help("How many times to subdivide").scan<'i', int>();
    program.add_argument("-nrf", "--normal_receptive_field").default_value(1).help("Receptive field of normal calculation (in px)").scan<'i', int>();
    program.parse_args(argc, argv);

    const int occlusion_boundary_line_width = program.get<int>("--line_width");

    const fs::path input_dir(program.get<std::string>("--input_dir"));
    const fs::path output_dir(program.get<std::string>("--output_dir"));
    if (input_dir.stem().string() == "x")
        exit(174); // Custom error code for checking if EGL is working
    assert_exists(input_dir);
    if (!fs::exists(output_dir))
        fs::create_directory(output_dir);

    const int frame_number = program.get<int>("--frame");
    const std::string frame_str = "frame_" + zfill(4, frame_number);
    const auto camera_dir = input_dir / frame_str / "cameras";
    assert_exists(camera_dir);
    std::vector<std::pair<fs::path, std::string>> camview_files;
    for (const auto &entry : fs::directory_iterator(camera_dir)){
        const auto matches = match_regex("camview_([0-9]+_[0-9]+_[0-9]+_[0-9]+)", entry.path().stem().string());
        const auto output_suffix = matches[1];
        MRASSERT(!matches.empty(), entry.path().string() + " did not match camview regex");
        camview_files.push_back({entry, output_suffix});
    }
    MRASSERT(!camview_files.empty(), "camview_files is empty");

    const auto image_shape = npz(camview_files[0].first).read_data<long>("HW");
    const int output_h = image_shape[0];
    const int output_w = image_shape[1];
    const int buffer_width = output_w * 2;
    const int buffer_height = output_h * 2;

    if ((buffer_width > 10000) || (buffer_height > 10000)){
        cout << "The image size [" << buffer_width << " x " << buffer_height << "] is too large." << endl;
        return 1;
    }

    // Linux (works headless)
    #if SYSTEM_NUM == 0

        static const EGLint pbufferAttribs[] = {
            EGL_WIDTH, buffer_width,
            EGL_HEIGHT, buffer_height,
            EGL_NONE,
        };

        EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

        EGLint major, minor;

        EGLBoolean res = eglInitialize(eglDpy, &major, &minor);
        assert (res != EGL_FALSE);

        // 2. Select an appropriate configuration
        EGLint numConfigs;
        EGLConfig eglCfg;

        eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

        // 3. Create a surface
        EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs); // eglSurf

        // 4. Bind the API
        eglBindAPI(EGL_OPENGL_API);

        // 5. Create a context and make it current
        const EGLint GiveMeGLES2[] = {
            EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL_CONTEXT_CLIENT_VERSION, 4,
            EGL_CONTEXT_CLIENT_VERSION, 4,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, 
            EGL_NONE
        };

        EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);//GiveMeGLES2);

        eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);

        GLADloadproc load = (GLADloadproc)eglGetProcAddress;
        if (!gladLoadGLLoader(load)){
            cerr << "Failed to initialize GLAD" << endl;
            return 188;
        }

    // MacOS, Windows?
    #elif SYSTEM_NUM == 0

        // glfw: initialize and configure
        // ------------------------------
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        #ifdef __APPLE__
        std::cout << "Apple!" << std::endl;
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif

        // glfw window creation
        // --------------------
        GLFWwindow* window = glfwCreateWindow(buffer_width, buffer_height, "LearnOpenGL", NULL, NULL);
        if (window == NULL)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return -1;
        }
        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        // glad: load all OpenGL function pointers
        // ---------------------------------------
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return -1;
        }
    #endif

    std::vector<CameraView> camera_views;
    for (const auto &entry : camview_files)
        camera_views.push_back({entry.second, camera_dir, output_w, output_h});

    const auto glsl = source_directory / "glsl";
    Shader spineShader((glsl / "wings.vert").c_str(), (glsl / "spine.frag").c_str(), (glsl / "spine.geom").c_str());
    Shader wingsShader((glsl / "wings.vert").c_str(), (glsl / "wings.frag").c_str(), (glsl / "wings.geom").c_str());
    Shader hairShader((glsl / "hair.vert").c_str(), (glsl / "hair.frag").c_str(), (glsl / "hair.geom").c_str());
    Shader nextShader((glsl / "next_wings.vert").c_str(), (glsl / "wings.frag").c_str(), (glsl / "wings.geom").c_str());

    glEnable(GL_DEPTH_TEST);
    glLineWidth(occlusion_boundary_line_width);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  

    GLint storage_buf_size;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &storage_buf_size);
    const size_t max_buf_size = (((size_t)storage_buf_size)+1) / (1024 * 1024);
    std::cout << "GL_MAX_SHADER_STORAGE_BLOCK_SIZE is " << max_buf_size << " MB." << endl;

    /*
        Drawing the 3D meshes. Instances are small-ish so we draw them all at once. Non-instances are
        larger so we draw them in chunks. Smaller stride -> smaller maxmimum memory.
    */
   std::vector<json> all_bboxes;
    while (true) {
        const auto some_object_model = load_blender_mesh(input_dir / frame_str / "mesh" / "saved_mesh.json");
        if (some_object_model == nullptr)
            break;

        if (some_object_model->type == Mesh)
            all_bboxes.push_back(some_object_model->bounding_box);

        for (const auto &cd : camera_views){

            // DEPTH
            glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer);
            cd.activateShader(wingsShader);

            if (some_object_model->type == Mesh)
                some_object_model->draw(wingsShader);

            cd.activateShader(hairShader);
            if (some_object_model->type == Hair)
                some_object_model->draw(hairShader);

            cd.activateShader(wingsShader);


            // OCCLUSION BOUNDARIES
            glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer_ob);

            if (some_object_model->type == Mesh)
                some_object_model->draw(wingsShader);

            cd.activateShader(hairShader);
            if (some_object_model->type == Hair)
                some_object_model->draw(hairShader);

            cd.activateShader(spineShader);

            if (some_object_model->type == Mesh)
                some_object_model->draw(spineShader);

            // NEXT FACE IDS
            glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer_next_faceids);
            cd.activateShader(nextShader);

            if (some_object_model->type == Mesh)
                some_object_model->draw(wingsShader);

        }
    }

    BaseBlenderObject::print_stats();

    for (const auto &cd : camera_views){
        glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer);

        /*
        Save bounding boxes
        */
        {
            std::ofstream o(output_dir / ("Objects_" + cd.frame_string + ".json"));
            o << json(all_bboxes).dump() << std::endl;
        }

        /*
            Reading/Writing the depth map
        */
       Eigen::Tensor<double, 3> flow3d;
        {
        auto pixels = read_buffer<float>(GL_COLOR_ATTACHMENT1, buffer_width, buffer_height);
        Eigen::Tensor<double, 2> depth(buffer_height, buffer_width);
        depth.setConstant(std::numeric_limits<double>::max());
        Eigen::Tensor<double, 3> points_3d(buffer_height, buffer_width, 3);
        points_3d.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying Depth")){
            if (pixels[o.j+2] > 0){
                for (int k=0; k<3; k++)
                    points_3d(o.y, o.x,k) = pixels[o.j+k];
                depth(o.y, o.x) = pixels[o.j+2];
            }
            o.progressbar();
        }
        auto depth_color = to_color_map(depth, 0.0, 0.90);
        imwrite(output_dir / ("Depth_" + cd.frame_string + ".png"), depth_color);
        save_npy(output_dir / ("Depth_" + cd.frame_string + ".npy"), depth);
        

        /*
            Reading/Writing the 3D optical flow
        */
        pixels = read_buffer<float>(GL_COLOR_ATTACHMENT2, buffer_width, buffer_height);
        Eigen::Tensor<double, 3> next_points_3d(buffer_height, buffer_width, 3);
        next_points_3d.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying Optical Flow")){
            if (pixels[o.j+2] > 0){
                for (int k=0; k<3; k++)
                    next_points_3d(o.y, o.x,k) = pixels[o.j+k];
            }
            o.progressbar();
        }
        flow3d = cd.project(next_points_3d) - cd.project(points_3d);
        const auto flow_viz = compute_flow_viz(flow3d);
        imwrite(output_dir / ("Flow3D_" + cd.frame_string + ".png"), flow_viz);
        save_npy(output_dir / ("Flow3D_" + cd.frame_string + ".npy"), flow3d);
        }

        /*
            Reading/Writing the geometry normal
        */
       {
        const auto pixels = read_buffer<float>(GL_COLOR_ATTACHMENT7, buffer_width, buffer_height);
        Eigen::Tensor<double, 3> geo_normals(output_h, output_w, 3);
        geo_normals.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Geometry Surface Normals")){
            for (int k=0; k<3; k++)
                geo_normals(o.y/2, o.x/2, k) += pixels[o.j+k];
            o.progressbar();
        }

        Eigen::Tensor<unsigned char, 3> geo_normal_color(output_h, output_w, 3);
        geo_normal_color.setZero();
        for (const loop_obj &o : image_iterator(output_w, output_h, "")){
            const double norm = std::pow(std::pow(geo_normals(o.y, o.x, 0), 2) + std::pow(geo_normals(o.y, o.x, 1), 2) + std::pow(geo_normals(o.y, o.x, 2), 2), 0.5);
            for (int k=0; k<3; k++){
                geo_normals(o.y, o.x, k) = geo_normals(o.y, o.x, k)/norm;
                geo_normal_color(o.y, o.x, k) = ((geo_normals(o.y, o.x, k) + 1) * (255/2));
            }
        }
        
        imwrite(output_dir / ("SurfaceNormal_" + cd.frame_string + ".png"), geo_normal_color);
        save_npy(output_dir / ("SurfaceNormal_" + cd.frame_string + ".npy"), geo_normals);
       }

        /*
            Reading/Writing the instance segmentation map
        */
        {
            const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT6, buffer_width, buffer_height);
            Eigen::Tensor<int, 3> instance_seg(buffer_height, buffer_width, 3);
            instance_seg.setZero();
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying instance segmentation masks")){
                for (int k=0; k<3; k++)
                    instance_seg(o.y, o.x, k) = pixels[o.j + k];
                o.progressbar();
            }
            save_npy(output_dir / ("InstanceSegmentation_" + cd.frame_string + ".npy"), instance_seg);

            Eigen::Tensor<long, 2> instance_seg_2d(buffer_height, buffer_width);
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height))
                instance_seg_2d(o.y, o.x) = long(instance_seg(o.y, o.x, 0) % 1000) + 1000 * long(instance_seg(o.y, o.x, 1) % 1000) + 1000000 * long(instance_seg(o.y, o.x, 2) % 1000);
            imwrite(output_dir / ("InstanceSegmentation_" + cd.frame_string + ".png"), to_color_map(instance_seg_2d));
        }

        /*
            Reading/Writing the object segmentation map
        */
        {
            const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT4, buffer_width, buffer_height);
            Eigen::Tensor<int, 2> object_seg(buffer_height, buffer_width);
            object_seg.setZero();
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying object segmentation masks")){
                object_seg(o.y, o.x) = pixels[o.j];
                o.progressbar();
            }
            save_npy(output_dir / ("ObjectSegmentation_" + cd.frame_string + ".npy"), object_seg);
            imwrite(output_dir / ("ObjectSegmentation_" + cd.frame_string + ".png"), to_color_map(object_seg.cast<long>()));
        }

        {
            const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT5, buffer_width, buffer_height);
            Eigen::Tensor<int, 2> tag_seg(buffer_height, buffer_width);
            tag_seg.setZero();
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying tag segmentation mask")){
                tag_seg(o.y, o.x) = pixels[o.j];
                o.progressbar();
            }
            save_npy(output_dir / ("TagSegmentation_" + cd.frame_string + ".npy"), tag_seg);
            imwrite(output_dir / ("TagSegmentation_" + cd.frame_string + ".png"), to_color_map(tag_seg.cast<long>()));
        }


        /*
            Reading/Writing the face id map
        */
        {
        Eigen::Tensor<int, 4> faceids(2, buffer_height, buffer_width, 3);
        faceids.setZero();
        {
            const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT3, buffer_width, buffer_height);
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying face ids from first frame")){
                for (int k=0; k<3; k++)
                    faceids(0, o.y, o.x, k) = pixels[o.j+k];
                o.progressbar();
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer_next_faceids);
        {
            const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT3, buffer_width, buffer_height);
            for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying face ids from second frame")){
                for (int k=0; k<3; k++)
                    faceids(1, o.y, o.x, k) = pixels[o.j+k];
                o.progressbar();
            }
        }
        Eigen::Array<unsigned char, -1, -1> flow_occlusion(buffer_height, buffer_width);
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Calculating flow occlusion")){
            const float fx = flow3d(o.y, o.x, 0);
            const float fy = flow3d(o.y, o.x, 1);
            bool match_exists = false;
            for (int i=-1; i<=1; i++){
                for (int j=-1; j<=1; j++){
                    const int next_x = o.x + lround(fx) + i;
                    const int next_y = o.y + lround(fy) + j;
                    if ((0 <= next_y) && (next_y < buffer_height) && (0 <= next_x) && (next_x < buffer_width)){
                        const bool face_ids_match = (
                            (faceids(0, o.y, o.x, 0) == faceids(1, next_y, next_x, 0)) &&
                            (faceids(0, o.y, o.x, 1) == faceids(1, next_y, next_x, 1)) &&
                            (faceids(0, o.y, o.x, 2) == faceids(1, next_y, next_x, 2))
                        );
                        match_exists = match_exists || face_ids_match;
                    }
                }
            }
            flow_occlusion(o.y, o.x) = ((unsigned char)match_exists) * 255;
            o.progressbar();
        }
        imwrite(output_dir / ("Flow3DMask_" + cd.frame_string + ".png"), flow_occlusion);
        }

        /*
        //  Reading/Writing the face size in m^2
        read_buffer<float>(GL_COLOR_ATTACHMENT3, pixels, buffer_width, buffer_height);
        auto face_size = Eigen::Tensor<double, 2>(output_h,output_w);
        face_size.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying Face-Sizes (in m^2)")){
            face_size(o.y/2, o.x/2) = pixels[o.j]/4;
            o.progressbar();
        }
        auto face_size_color = to_color_map(face_size, 0, 0.5, 0);
        imwrite(output_dir / ("FaceSizeCM" + cd.frame_string + ".png"), face_size_color);

        //    Reading/Writing the face size in pixels
        read_buffer<float>(GL_COLOR_ATTACHMENT4, pixels, buffer_width, buffer_height);
        auto pixel_size = Eigen::Tensor<double, 2>(output_h,output_w);
        pixel_size.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying Face-Sizes (in pixels)")){
            pixel_size(o.y/2, o.x/2) = pixels[o.j]/4;
            o.progressbar();
        }
        auto pixel_size_color = to_color_map(pixel_size, 0, 0.5, 0);
        imwrite(output_dir / ("FaceSizePX" + cd.frame_string + ".png"), pixel_size_color);

        */

        glBindFramebuffer(GL_FRAMEBUFFER, cd.framebuffer_ob);

        /*
            Reading/Writing the occlusion boundaries
        */
       {
        const auto pixels = read_buffer<int>(GL_COLOR_ATTACHMENT0, buffer_width, buffer_height);
        Eigen::Array<unsigned char, -1, -1> occlusion_boundaries(buffer_height,buffer_width);
        occlusion_boundaries.setZero();
        for (const loop_obj &o : image_iterator(buffer_width, buffer_height, "Copying Occlusion Boundaries")){
            occlusion_boundaries(o.y, o.x) = pixels[o.j+1]*255;
            o.progressbar();
        }
        imwrite(output_dir / ("OcclusionBoundaries_" + cd.frame_string + ".png"), occlusion_boundaries);
       }

    }
    cout << "Done." << endl;
}
