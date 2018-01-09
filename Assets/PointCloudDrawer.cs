using OpenCVForUnity;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class PointCloudDrawer : MonoBehaviour
{
    private const float near_plane = 0.01f;
    private const float far_plane = 200f;
    private const float sphere_size = 0.008f;

    [SerializeField]
    private string directory_to_png;

    private int current_frame = 0;

    private Texture2D _texture2D;
    private Dictionary<long, Matrix4x4> dict =
        new Dictionary<long, Matrix4x4>();

    private List<string> _images;
    private GameObject _backgroundMesh;

    [SerializeField]
    private Material _material;

    [SerializeField]
    private Camera _camera;

    private static CameraIntrinsics GetIntrinsics()
    {
        return new CameraIntrinsics
        (
            width: 1280,
            height: 720,

            fx: 1028.0f,
            fy: 1028.0f,
            cx: 640.0f,
            cy: 360.0f
        );
    }

    private static DistortionCoefficient GetDistortionCoef()
    {
        return new DistortionCoefficient()
        {
            k1 = 0.0f,
            k2 = 0.0f,
            p1 = 0.0f,
            p2 = 0.0f,
            k3 = 0.0f
        };
    }

    Matrix4x4 YAxisInversion =
        Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));

    private GameObject _root;

    private Mat _camMatrix;

    private uint _width;
    private uint _height;

    public class PairOfMatrixAtFrame
    {
        public Matrix4x4 matrix_slam;
        public Matrix4x4 matrix_aruco;
    }

    private Dictionary<int, PairOfMatrixAtFrame> _pairOfMatrices =
        new Dictionary<int, PairOfMatrixAtFrame>();

    private float _markerLength;
    private Mat _ids;
    private List<Mat> _corners;
    private List<Mat> _rejected;
    private Mat _rvecs;
    private Mat _tvecs;
    private DetectorParameters _detectorParams;
    private Dictionary _dictionary;
    private MatOfDouble _distCoeffs;
    
    private IntPtr _slamSystem = IntPtr.Zero;
    private IntPtr _vocabFile;
    private float[] _matrix = new float[16];
    private float _scale_factor;
    private bool _has_computed_scale_factor;
    private Matrix4x4 _matrixFromSlamToAruco;
    private bool _has_matrix;
    private bool _hasCreatePoints;

    void Start()
    {
        var intrin = GetIntrinsics();
        _width = intrin.Width;
        _height = intrin.Height;
        GetPngs();
        InitAruco();
        ApplyProjectionMatrix();
        CreateSlamSystem();
    }
    
    private void OnGUI()
    {
        GUI.Label(new UnityEngine.Rect(0, 0, 200, 200), "frame is " + current_frame);
    }

    private void InitAruco()
    {
        var dictionaryId = 10;
        _markerLength = 0.1f;

        _ids = new Mat();
        _corners = new List<Mat>();
        _rejected = new List<Mat>();
        _rvecs = new Mat();
        _tvecs = new Mat();

        _detectorParams = DetectorParameters.create();
        _dictionary = Aruco.getPredefinedDictionary(dictionaryId);

        var intrin = GetIntrinsics();

        _camMatrix = new Mat(3, 3, CvType.CV_64FC1);
        _camMatrix.put(0, 0, intrin.Fx);
        _camMatrix.put(0, 1, 0);
        _camMatrix.put(0, 2, intrin.Cx);
        _camMatrix.put(1, 0, 0);
        _camMatrix.put(1, 1, intrin.Fy);
        _camMatrix.put(1, 2, intrin.Cy);
        _camMatrix.put(2, 0, 0);
        _camMatrix.put(2, 1, 0);
        _camMatrix.put(2, 2, 1.0f);

        var dist = GetDistortionCoef();
        _distCoeffs = new MatOfDouble(dist.k1, dist.k2, dist.p1, dist.p2);
    }

    private void CreateSlamSystem()
    {
        var vocabFilePath = Path.Combine(Application.streamingAssetsPath, "vocabulary.bin");
        var vocalFilesPathsAsBytes = Encoding.ASCII.GetBytes(vocabFilePath);

        var cameraConfigFile = Path.Combine(Application.streamingAssetsPath, "cameraConfig.yaml");
        var cameraConfigFiles = Encoding.ASCII.GetBytes(cameraConfigFile);

        var handle1 = GCHandle.Alloc(vocalFilesPathsAsBytes);
        var handle2 = GCHandle.Alloc(cameraConfigFiles);

        var isDisplayingWindow = false;

        byte displayWindowAsByte =
            isDisplayingWindow ?
                (byte)1 :
                (byte)0;

        _vocabFile = SlamWrapper.read_vocab_file(ref vocalFilesPathsAsBytes[0]);

        _slamSystem =
            SlamWrapper.create_SLAM_system(
                _vocabFile,
                ref cameraConfigFiles[0],
                displayWindowAsByte
        );

        handle1.Free();
        handle2.Free();
    }

    private void GetPngs()
    {
        if (!Directory.Exists(directory_to_png))
        {
            var msg = "Directory " + directory_to_png + " does not exist";
            Debug.Log(msg);
            throw new Exception(msg);
        }

        _images = Directory.GetFiles(directory_to_png, "*.png").ToList();
        var realPath = Path.Combine(directory_to_png, "image-");
        var length = realPath.Length;

        Comparison<string> comparison =
            new Comparison<string>
            (
                (a, b) =>
                {
                    var a_p = a.Remove(0, length).Replace(".png", "");
                    var b_p = b.Remove(0, length).Replace(".png", "");
                    var a_nb = Int32.Parse(a_p);
                    var b_nb = Int32.Parse(b_p);
                    return a_nb - b_nb;
                }
            );

        _images.Sort(comparison);

        var intrinsics = GetIntrinsics();
        var distCoefs = GetDistortionCoef();

        int width = (int)intrinsics.Width;
        int height = (int)intrinsics.Height;

        int halfWidth = width / 2;
        int halfHeight = height / 2;

        _texture2D = new Texture2D(width, height, TextureFormat.ARGB32, false);
        _backgroundMesh = GameObject.CreatePrimitive(PrimitiveType.Quad);
        _backgroundMesh.transform.parent = _camera.transform;
        var renderer = _backgroundMesh.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Custom/undistordShader"));

        mat.SetFloat("_Fx", intrinsics.Fx);
        mat.SetFloat("_Fy", intrinsics.Fy);
        mat.SetFloat("_Cx", intrinsics.Cx);
        mat.SetFloat("_Cy", intrinsics.Cy);

        mat.SetFloat("_K1", distCoefs.k1);
        mat.SetFloat("_K2", distCoefs.k2);
        mat.SetFloat("_P1", distCoefs.p1);
        mat.SetFloat("_P2", distCoefs.p2);
        mat.SetFloat("_K3", distCoefs.k3);

        mat.SetFloat("_Width", intrinsics.Width);
        mat.SetFloat("_Height", intrinsics.Height);

        renderer.material = mat;
        _backgroundMesh.GetComponent<Renderer>().material.mainTexture = _texture2D;

        var aspect_ratio = (float)width / height;

        var frustum =
            BuildFrustumFromIntrinsics
            (
                intrinsics,
                near_plane,
                far_plane
            );

        _backgroundMesh.transform.localScale =
            new Vector3
            (
                x: (frustum.Top - frustum.Bottom) * aspect_ratio,
                y: (frustum.Top - frustum.Bottom),
                z: 1.0f
            );

        var frustumSizeHorizontal = frustum.Right - frustum.Left;
        var frustumSizeVertical = frustum.Top - frustum.Bottom;

        var deltaXPercent = (intrinsics.Cx - halfWidth) / width;
        var deltaYPercent = (intrinsics.Cy - halfHeight) / height;

        _backgroundMesh.transform.localPosition =
            new Vector3
            (
                deltaXPercent * frustumSizeHorizontal,
                deltaYPercent * frustumSizeVertical,
                near_plane + 0.00001f
            );
    }

    private void CreatePointCloudFromVector3dArray(Vector3[] positions)
    {
        _root = new GameObject();
        _root.transform.position = Vector3.zero;
        _root.transform.rotation = Quaternion.identity;
        _root.name = "Point cloud";

        int i = 0;
        foreach (var pos in positions)
        {
            var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obj.transform.parent = _root.transform;
            obj.transform.position = pos;
            obj.transform.localScale = Vector3.one * sphere_size;
            obj.name = "point_" + i.ToString("00000");
            i++;
        }
    }

    public Matrix4x4 ChangeReferenceOfMatrix(Matrix4x4 m)
    {
        var a = YAxisInversion;
        var b = a.inverse;
        return b * m * a;
    }

    private void OnDestroy()
    {
        if (_slamSystem.ToInt64() != 0)
        {
            SlamWrapper.shutdown_slam_system(_slamSystem);
            SlamWrapper.delete_pointer(_slamSystem);
            _slamSystem = IntPtr.Zero;
        }

        if (_vocabFile.ToInt64() != 0)
        {
            SlamWrapper.delete_pointer(_vocabFile);
            _vocabFile = IntPtr.Zero;
        }
    }

    void Update()
    {
        Matrix4x4 maybe_slam_matrix = Matrix4x4.identity;
        bool is_slam_matrix_set = false;

        var stopwatchTex = new System.Diagnostics.Stopwatch();
        stopwatchTex.Start();
        _texture2D.LoadImage(File.ReadAllBytes(_images[current_frame]));
        _texture2D.Apply();
        stopwatchTex.Stop();
        //Debug.LogWarning("elapsed time for texture reading " + stopwatchTex.ElapsedMilliseconds + " ms");

        Mat rgbMat = new Mat((int)_height, (int)_width, CvType.CV_8UC3);
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Utils.texture2DToMat(_texture2D, rgbMat);
        //Debug.LogWarning("elapsed time for texture conversion is " + stopwatch.ElapsedMilliseconds + " ms");

        Matrix4x4 aruco_mat = Matrix4x4.identity;
        var has_detected = false;

        var thread = new System.Threading.Thread(() =>
        {
            var stopwatchAruco = new System.Diagnostics.Stopwatch();
            stopwatchAruco.Start();
            has_detected = RunAruco(rgbMat, ref aruco_mat);
            stopwatchAruco.Stop();
            //Debug.LogWarning("elapsed time for running aruco is " + stopwatchAruco.ElapsedMilliseconds + " ms");

        });

        thread.Start();
        
        var current_index = current_frame;
        var stopwatchSlam = new System.Diagnostics.Stopwatch();
        stopwatchSlam.Start();
        RunSlam(ref maybe_slam_matrix, ref is_slam_matrix_set, rgbMat);
        stopwatchSlam.Stop();
        //Debug.LogWarning("elapsed time for running slam is " + stopwatchSlam.ElapsedMilliseconds + " ms");

        thread.Join();
        var slam_matrix = Matrix4x4.identity;

        if (is_slam_matrix_set)
        {
            slam_matrix = ToUnityFrame(maybe_slam_matrix);
            _scale_factor = 1.0f;
            _has_computed_scale_factor = true;
        }

        if (has_detected)
        {
            //ApplyPoseMatrix(aruco_matrix);
            if (!_pairOfMatrices.ContainsKey(current_index))
            {
                _pairOfMatrices.Add(current_index, new PairOfMatrixAtFrame()
                {
                    matrix_aruco = aruco_mat,
                    matrix_slam = slam_matrix
                });
            }

            if (is_slam_matrix_set && _has_computed_scale_factor && !_has_matrix)
            {
                var scaleFactor = _scale_factor;

                _matrixFromSlamToAruco =
                    ComputeSlamToArucoMatrix(slam_matrix, aruco_mat, scaleFactor);

                Debug.LogWarning("matrix from slam to aruco computed.");

                _has_matrix = true;
            }

            if (!_hasCreatePoints && is_slam_matrix_set && _has_matrix)
            {
                int array_size = 0;

                var array_ptr =
                    SlamWrapper.get_3d_tracked_points(_slamSystem, ref array_size);

                if (array_ptr.ToInt64() != 0 && array_size > 0)
                {
                    var tracked3DPoints = new float[array_size];
                    Marshal.Copy(array_ptr, tracked3DPoints, 0, array_size);
                    SlamWrapper.delete_pointer(array_ptr);

                    var cloud_points_3d = new List<Vector3>();
                    for (int i = 0; i < tracked3DPoints.Length / 4; i++)
                    {
                        var x = tracked3DPoints[i * 4 + 0];
                        var y = tracked3DPoints[i * 4 + 1];
                        var z = tracked3DPoints[i * 4 + 2];
                        var isTracked = tracked3DPoints[i * 4 + 3] != 0 ? true : false;

                        var p = new Vector4(x, y, z, 1);

                        var m = Matrix4x4.TRS(p, Quaternion.identity, Vector3.one);
                        var mat = ChangeReferenceOfMatrix(m);
                        var pos_ = mat.GetColumn(3);
                        pos_ = pos_ / pos_.w;

                        var pos = _matrixFromSlamToAruco * pos_;

                        pos = pos / pos.w;
                        cloud_points_3d.Add(pos);
                    }

                    Debug.LogWarning("creating point cloud");
                    CreatePointCloudFromVector3dArray(cloud_points_3d.ToArray());
                    _hasCreatePoints = true;

                    var result = _matrixFromSlamToAruco * slam_matrix;
                    ApplyPoseMatrix(result);

                    var marker_points = ArucoTagDict10Id1PointList.GetPoints();

                    // one block in the tag is 1.2 cm long.
                    var markers_points_in_meters =
                        marker_points.Select(pt => pt * 0.012f).ToList();

                    var markers_points_screen_frame =
                        markers_points_in_meters.Select(pt =>
                        {
                            var pt_3d = new Vector3(pt.x, pt.y, 0);
                            var proj = _camera.WorldToScreenPoint(pt_3d);
                            return new Vector2(proj.x, proj.y);
                        }).ToList();

                    var cloud_points_screen_frame =
                        cloud_points_3d.Select(pt =>
                        {
                            var proj = _camera.WorldToScreenPoint(pt);
                            return new Vector2(proj.x, proj.y);
                        }).ToList();

                    var dict = new Dictionary<int, int>();

                    for (int i = 0; i < marker_points.Count; i++)
                    {
                        var pt_2d = marker_points[i];
                        var pt_2d_projected = markers_points_screen_frame[i];
                        
                        var index = 
                            FindClosest(pt_2d_projected, cloud_points_screen_frame);

                        dict.Add(i, index);
                    }
                    
                    var camera_position = _camera.transform.position;
                    var keys = dict.Keys;
                    var scale_factor_values = new List<float>();

                    foreach(var key in keys)
                    {
                        var i = key;
                        var index = dict[i];
                        var point3d = cloud_points_3d[index];
                        var delta = point3d - camera_position;
                        var k = -camera_position.z / delta.z;
                        scale_factor_values.Add(k);

                        //var pt_2d = marker_points[i];
                        //var pt_2d_in_m = markers_points_in_meters[i];
                        //var pt_2d_projected = markers_points_screen_frame[i];
                        //var point3d_projected = 
                        //    cloud_points_screen_frame[index];
                        //var distance = Vector2.Distance(pt_2d_projected, point3d_projected);
                        //var height = (int) GetIntrinsics().Height;
                        //Debug.DrawLine(pt_2d_in_m, point3d);
                    }

                    _scale_factor = scale_factor_values.Average();

                    _matrixFromSlamToAruco = ComputeSlamToArucoMatrix(slam_matrix, aruco_mat, _scale_factor);
                    _has_matrix = true;

                    for (int i = 0; i < _root.transform.childCount; i++)
                    {
                        var current_child = _root.transform.GetChild(i);
                        current_child.transform.position /= _scale_factor;
                    }
                    //Debug.LogError("pause");
                }
            }
        }

        if (is_slam_matrix_set && _has_matrix)
        {
            // transform camera pose from Slam reference frame
            // to the one in the Aruco one
            var result = _matrixFromSlamToAruco * slam_matrix;
            ApplyPoseMatrix(result);
        }

        if (Time.frameCount % 1 == 0)
        {
            current_frame++;
            current_frame = current_frame % _images.Count;
        }
    }

    private static int FindClosest
    (
        Vector2 pt_2d_projected, 
        List<Vector2> cloud_points_screen_frame
    )
    {
        var dist = 2.0f * Vector2.Distance(pt_2d_projected, cloud_points_screen_frame[0]);
        var best_index = -1;
        for (int i = 0; i < cloud_points_screen_frame.Count; i++)
        {
            var current_cloud_point_screen_frame = cloud_points_screen_frame[i];
            var current_dist = Vector2.Distance(pt_2d_projected, current_cloud_point_screen_frame);

            if (current_dist < dist)
            {
                dist = current_dist;
                best_index = i;
            }
        }
        return best_index;
    }

    private bool RunAruco(Mat rgbMat, ref Matrix4x4 aruco_matrix)
    {
        var hasDetected = ArucoDetection(rgbMat, out aruco_matrix);
        return hasDetected;
    }

    private void RunSlam(ref Matrix4x4 maybe_slam_matrix, ref bool is_slam_matrix_set, Mat rgbMat)
    {
        IntPtr pose =
            SlamWrapper.update_image
            (
                _slamSystem,
                new IntPtr(rgbMat.dataAddr()),
                (int)_width,
                (int)_height,
                Time.timeSinceLevelLoad
            );

        if (pose.ToInt64() != 0)
        {
            Marshal.Copy(pose, _matrix, 0, 16);
            SlamWrapper.delete_pointer(pose);

            for (int i = 0; i < 16; i++)
            {
                maybe_slam_matrix[i] = _matrix[i];
            }

            is_slam_matrix_set = true;
            maybe_slam_matrix = maybe_slam_matrix.transpose;
        }
    }

    private Matrix4x4 ComputeSlamToArucoMatrix
    (
        Matrix4x4 mat_slam,
        Matrix4x4 mat_aruco,
        float scaleFactor
    )
    {

        var fix_scale =
            Matrix4x4.TRS
            (
                Vector3.zero,
                Quaternion.identity,
                Vector3.one * scaleFactor
            );

        var p = mat_aruco * fix_scale * mat_slam.inverse;

        // Extract new local position
        Vector3 position = p.GetColumn(3);

        // Extract new local rotation
        Quaternion rotation = Quaternion.LookRotation(
            p.GetColumn(2),
            p.GetColumn(1)
        );

        // Extract new local scale
        Vector3 scale = new Vector3(
            p.GetColumn(0).magnitude,
            p.GetColumn(1).magnitude,
            p.GetColumn(2).magnitude
        );

        //Debug.LogWarning(position.x + " " + position.y + " " + position.z);
        //Debug.LogWarning(rotation.x + " " + rotation.y + " " + rotation.z + " " + rotation.w);
        //Debug.LogWarning(scale.x + " " + scale.y + " " + scale.z);
        return p;
    }


    private bool ArucoDetection(Mat rgbMat, out Matrix4x4 mat)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Aruco.detectMarkers(rgbMat, _dictionary, _corners, _ids, _detectorParams, _rejected);
        //Debug.LogWarning("elapsed time for marker detection is " + stopwatch.ElapsedMilliseconds + " ms");
        if (_ids.total() > 0)
        {
            stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();
            Aruco.estimatePoseSingleMarkers(_corners, _markerLength, _camMatrix, _distCoeffs, _rvecs, _tvecs);
            //Debug.LogWarning("elapsed time for marker pose retrieval is " + stopwatch.ElapsedMilliseconds + " ms");
            var mat2 = GetMatrix(_tvecs.get(0, 0), _rvecs.get(0, 0));
            mat = mat2;
            Aruco.drawAxis(rgbMat, _camMatrix, _distCoeffs, _rvecs, _tvecs, _markerLength * 0.5f);
            return true;
        }
        mat = Matrix4x4.identity;
        return false;
    }

    private Matrix4x4 GetMatrix(double[] tvec, double[] rv)
    {
        Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);
        Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
        rvec.put(0, 0, rv[0]);
        rvec.put(1, 0, rv[1]);
        rvec.put(2, 0, rv[2]);
        Calib3d.Rodrigues(rvec, rotMat);

        Matrix4x4 transformationM = new Matrix4x4(); // from OpenCV
        transformationM.SetRow(0, new Vector4((float)rotMat.get(0, 0)[0], (float)rotMat.get(0, 1)[0], (float)rotMat.get(0, 2)[0], (float)tvec[0]));
        transformationM.SetRow(1, new Vector4((float)rotMat.get(1, 0)[0], (float)rotMat.get(1, 1)[0], (float)rotMat.get(1, 2)[0], (float)tvec[1]));
        transformationM.SetRow(2, new Vector4((float)rotMat.get(2, 0)[0], (float)rotMat.get(2, 1)[0], (float)rotMat.get(2, 2)[0], (float)tvec[2]));
        transformationM.SetRow(3, new Vector4(0, 0, 0, 1));
        Matrix4x4 invertZM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));
        Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
        Matrix4x4 ARM = invertYM * transformationM;
        ARM = ARM * invertZM;
        return ARM.inverse;
    }

    private Matrix4x4 ToUnityFrame(Matrix4x4 m)
    {
        return ChangeReferenceOfMatrix(m).inverse;
    }

    private void ApplyProjectionMatrix()
    {
        CameraIntrinsics intrinsics = GetIntrinsics();
        var mat = BuildProjMatrixFromIntrinsics(intrinsics, near_plane, far_plane);
        _camera.projectionMatrix = mat;
    }

    private void ApplyPoseMatrix(Matrix4x4 m)
    {
        Vector3 t = m.GetColumn(3);

        Quaternion q =
            Quaternion.LookRotation(
                 m.GetColumn(2),
                 m.GetColumn(1)
            );

        _camera.transform.position = t;
        _camera.transform.rotation = q;
    }

    public static Matrix4x4 BuildProjMatrixFromIntrinsics
    (
        CameraIntrinsics intrinsics,
        float near,
        float far
    )
    {
        Matrix4x4 mat = Matrix4x4.zero;
        mat.m00 = 2.0f * intrinsics.Fx / intrinsics.Width;
        mat.m11 = 2.0f * intrinsics.Fy / intrinsics.Height;

        mat.m02 = 2.0f * (intrinsics.Cx / intrinsics.Width) - 1.0f;
        mat.m12 = 2.0f * (intrinsics.Cy / intrinsics.Height) - 1.0f;
        mat.m22 = -(far + near) / (far - near);
        mat.m32 = -1;

        mat.m23 = 2.0f * far * near / (near - far);
        return mat;
    }

    public struct Frustum
    {
        public float Left;
        public float Right;
        public float Bottom;
        public float Top;
        public float Near;
        public float Far;
    }

    public static Frustum BuildFrustumFromIntrinsics
    (
        CameraIntrinsics intrinsics,
        float near,
        float far
    )
    {
        float left = -intrinsics.Cx / intrinsics.Fx;
        float right = (intrinsics.Width - intrinsics.Cx) / intrinsics.Fx;
        float top = intrinsics.Cy / intrinsics.Fy;
        float bottom = -(intrinsics.Height - intrinsics.Cy) / intrinsics.Fy;

        return new Frustum()
        {
            Left = left * near,
            Right = right * near,
            Top = top * near,
            Bottom = bottom * near,
            Near = near,
            Far = far
        };
    }

    public struct CameraIntrinsics
    {
        public float Fx;
        public float Fy;
        public float Cx;
        public float Cy;

        public uint Width;
        public uint Height;

        public CameraIntrinsics
        (
            uint width,
            uint height,
            float fx,
            float fy,
            float cx,
            float cy
        )
        {
            Width = width;
            Height = height;
            Fx = fx;
            Fy = fy;
            Cx = cx;
            Cy = cy;
        }
    }

    public class DistortionCoefficient
    {
        public float k1;
        public float k2;
        public float p1;
        public float p2;
        public float k3;
    }

    public static class ArucoTagDict10Id1PointList
    {
        // points of the tag 1 of the aruco dictionary 10 for size of 6x6 bits.
        // https://docs.opencv.org/3.2.0/d5/dae/tutorial_aruco_detection.html
        // https://docs.opencv.org/3.2.0/marker23.jpg
        /*
        
        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
        MMMMMMMMMMMMMMMMMMMMMMy-------:yMMMM
        MMMMMMMMMMMMMMMMMMMMMMo        oMMMM
        MMMMhooooMMMMhoooooooo-        oMMMM
        MMMMo    MMMMo                 oMMMM
        MMMMo    MMMMo             ::::yMMMM
        MMMMo    MMMMo             MMMMMMMMM
        MMMMo    MMMMo             MMMMMMMMM
        MMMMo    MMMMMMMMMMMMMo        oMMMM
        MMMMo    MMMMMMMMMMMMMo        oMMMM
        MMMMo    MMMMMMMMMMMMMo    ddddmMMMM
        MMMMo    MMMMMMMMMMMMMo    MMMMMMMMM
        MMMMdoooooooohMMMMMMMMhoooooooohMMMM
        MMMMMMMMM    oMMMMMMMMMMMMM    oMMMM
        MMMMMMMMM::::yMMMMMMMMMMMMM::::yMMMM
        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

        */
        public static List<Vector2> GetPoints()
        {
            return new List<Vector2>()
            {
                // 1-5
                new Vector2(-3, 2),
                new Vector2(-2, 2),
                new Vector2(-3, -2),
                new Vector2(-2, -2),
                new Vector2(-1, -2),

                // 6-10
                new Vector2(-1, -3),
                new Vector2(-2, -3),
                new Vector2(2, -3),
                new Vector2(3, -3),
                new Vector2(3, -2),

                //11-15
                new Vector2(2, -2),
                new Vector2(2, -1),
                new Vector2(3, -1),
                new Vector2(3, 0),
                new Vector2(2, 0),

                //16-20
                new Vector2(2, 1),
                new Vector2(3, 1),
                new Vector2(3, 3),
                new Vector2(1, 3),
                new Vector2(1, 2),

                //21-24
                new Vector2(-1, 2),
                new Vector2(-1, 0),
                new Vector2(1, 0),
                new Vector2(1, 2),

            };
        }
    }
}
