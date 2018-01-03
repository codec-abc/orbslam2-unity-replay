﻿#define USE_RECORDED_DATA
#define USE_MACHINE_HALL

using OpenCVForUnity;
using OpenCVForUnityExample;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

public class PointCloudDrawer : MonoBehaviour
{

    [SerializeField]
    private string directory_to_png;

    private int current_frame = 204;
    private Texture2D _texture2D;
    private Dictionary<long, Matrix4x4> dict = new Dictionary<long, Matrix4x4>();
    private List<string> _images;
    private GameObject _backgroundMesh;

    [SerializeField]
    private Material _material;

    [SerializeField]
    private Camera _camera;

    private const float near_plane = 0.01f;
    private const float far_plane = 200f;
    private const float sphere_size = 0.008f;

    private static CameraIntrinsics GetIntrinsics()
    {
#if USE_MACHINE_HALL
        //return new CameraIntrinsics
        //(
        //    width: 752,
        //    height: 480,
        //    fx: 458.654f,
        //    fy: 457.296f,
        //    cx: 367.215f,
        //    cy: 248.375f
        //);

        return new CameraIntrinsics
        (
            width: 1280,
            height: 720,

            //fx: 1024.45f,
            //fy: 1031.36f,
            //cx: 673.26f,
            //cy: 438.08f

            fx: 1028.0f,
            fy: 1028.0f,
            cx: 640.0f,
            cy: 360.0f
        );
#else 
        return new CameraIntrinsics
        (
            width: 1280,
            height: 720,

            //fx: 1024.45f,
            //fy: 1031.36f,
            //cx: 673.26f,
            //cy: 438.08f

            fx: 1028.0f,
            fy: 1028.0f,
            cx: 640.0f,
            cy: 360.0f
        );
#endif
    }

    private static DistortionCoefficient GetDistortionCoef()
    {
#if USE_MACHINE_HALL
        //return new DistortionCoefficient()
        //{
        //    k1 = -0.28340811f,
        //    k2 = 0.07395907f,
        //    p1 = 0.00019359f,
        //    p2 = 1.76187114e-05f,
        //    k3 = 0
        //};

        return new DistortionCoefficient()
        {
            k1 = 0.0f,
            k2 = 0.0f,
            p1 = 0.0f,
            p2 = 0.0f,
            k3 = 0.0f
        };
#else
        return new DistortionCoefficient()
        {
            k1 = 0.0f,
            k2 = 0.0f,
            p1 = 0.0f,
            p2 = 0.0f,
            k3 = 0.0f
        };
#endif
    }

    Matrix4x4 YAxisInversion = 
        Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));

    private GameObject _root;

    [SerializeField]
    private bool _bypassUpdate = false;
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

#if !USE_RECORDED_DATA
    private IntPtr _slamSystem;
    private float[] _matrix = new float[16];
    private byte[] _fixed_buffer2;
    private byte[] _fixed_buffer;
    private uint _buffer_length;
    private uint _new_buffer_length;
    private GCHandle _pinnedArray;
    private IntPtr _pointer;
    private byte[] _managedArray;
#endif

    void Start()
    {
        var intrin = GetIntrinsics();
        _width = intrin.Width;
        _height = intrin.Height;
#if USE_RECORDED_DATA
        CreatePointCloud();
        RetrivePoses();
#endif
        GetPngs();
        InitAruco();
#if !USE_RECORDED_DATA
        CreateSlamSystem();
        File.Delete(@"C:\Users\sesa455926\Desktop\movieSlam2\poses.txt");
#endif
        ApplyProjectionMatrix();

        var go = GameObject.Find("Cube");

        var mat = 
            Matrix4x4.TRS
            (
                new Vector3(-0.1182f, -0.0089f, 0.8266f),
                Quaternion.Euler(-21.14f, 15.26f, -7.487f),
                Vector3.one
            );

        var inverse_mat = mat.inverse;

        //ARUtils.SetTransformFromMatrix(go.transform, ref mat);
        //go.transform.localScale = new Vector3(0.1635753f, 0.1635753f, 0.001f);

        ARUtils.SetTransformFromMatrix(_root.transform, ref inverse_mat);
        _root.transform.localScale = Vector3.one / (0.1635753f / 0.1f);
        _root.transform.localPosition = _root.transform.localPosition / 1.6335753f;

        for (int i = 0; i < _root.transform.childCount; i++)
        {
            var current_child = _root.transform.GetChild(i);
            current_child.localScale = sphere_size / 1.635753f * Vector3.one;
        }
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

#if !USE_RECORDED_DATA
    private void CreateSlamSystem()
    {
        var vocabFilePath = Path.Combine(Application.streamingAssetsPath, "ORBvoc.txt");
        var vocalFilesPathsAsBytes = Encoding.ASCII.GetBytes(vocabFilePath);

        var cameraConfigFile = Path.Combine(Application.streamingAssetsPath, "cameraConfig.yaml");
        var cameraConfigFiles = Encoding.ASCII.GetBytes(cameraConfigFile);

        var handle1 = GCHandle.Alloc(vocalFilesPathsAsBytes);
        var handle2 = GCHandle.Alloc(cameraConfigFiles);

        var isDisplayingWindow = true;
        byte displayWindowAsByte =
            isDisplayingWindow ?
                (byte)1 :
                (byte)0;

        var vocabFile = SlamWrapper.read_vocab_file(ref vocalFilesPathsAsBytes[0]);

        _slamSystem =
            SlamWrapper.create_SLAM_system(
                vocabFile,
                ref cameraConfigFiles[0],
                displayWindowAsByte
        );

        handle1.Free();
        handle2.Free();

        var intrin = GetIntrinsics();

        _fixed_buffer2 = new byte[_width * _height * 3];
        _fixed_buffer = new byte[_width * _height * 3];
        _buffer_length = _width * _height * 4;
        _new_buffer_length = _width * _height * 3;
        _pinnedArray = GCHandle.Alloc(_fixed_buffer, GCHandleType.Pinned);
        _pointer = _pinnedArray.AddrOfPinnedObject();
        _managedArray = new byte[_buffer_length];
    }
#endif

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

        int width = (int) intrinsics.Width;
        int height = (int) intrinsics.Height;

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

        var aspect_ratio = (float) width / height;

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

    private void CreatePointCloud()
    {
        var mapPointsFilePath =
            Path.Combine(Application.streamingAssetsPath, "mappoints.txt");

        var lines = File.ReadAllLines(mapPointsFilePath);

        var pos =
            lines
            .Where(str => str.Contains(' '))
            .Select(str => ToVector3(str))
            .ToArray();

        _root = new GameObject();
        _root.transform.position = Vector3.zero;
        _root.transform.rotation = Quaternion.identity;
        _root.name = "Point cloud";

        foreach (var a_pos in pos)
        {
            var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            obj.transform.parent = _root.transform;
            var m = Matrix4x4.TRS(a_pos, Quaternion.identity, Vector3.one);
            var new_mat = ChangeReferenceOfMatrix(m);
            var pos_4 = new_mat.GetColumn(3);
            var new_pos = new Vector3(pos_4.x, pos_4.y, pos_4.z);
            obj.transform.position = new_pos;
            obj.transform.localScale = Vector3.one * sphere_size;
        }
    }

    private Vector3 ToVector3(string str)
    {
        var splits = str.Split(' ');
        return new Vector3(
                x : float.Parse(splits[0]),
                y : float.Parse(splits[1]),
                z : float.Parse(splits[2])
            );
    }

    private void RetrivePoses()
    {
        var poseFilePath =
            Path.Combine(Application.streamingAssetsPath, "poses.txt");

        var lines = File.ReadAllLines(poseFilePath);
        long current_pose = 0;

        for (int i = 0; i < lines.Length;)
        {
            var current_line = lines[i];
            if (current_line.Contains("pose is"))
            {
                current_pose = Int64.Parse(current_line.Split(' ')[0]);
                i++;
            }
            else if (current_line.Contains("[]"))
            {
                dict.Add(current_pose, Matrix4x4.zero);
                i++;
            }
            else if (current_line.Contains("["))
            {
                var line0 = lines[i + 0].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line1 = lines[i + 1].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line2 = lines[i + 2].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");
                var line3 = lines[i + 3].Replace("[", "").Replace("]", "").Replace(" ", "").Replace(";", "");

                var mat = GetMatrix(line0, line1, line2, line3);
                dict.Add(current_pose, mat);
#if USE_MACHINE_HALL
                i += 4;
#else
                i += 5;
#endif
            }
            else
            {
                throw new Exception("something went wrong for line " + i + "ie " + current_line);
            }
        }
    }

    private Matrix4x4 GetMatrix
    (
        string line1, 
        string line2, 
        string line3, 
        string line4
    )
    {
#if USE_MACHINE_HALL
        var numbersLine1 = line1.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine2 = line2.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine3 = line3.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine4 = line4.Split(',').Select(str => float.Parse(str)).ToArray();
#else
        var numbersLine1 = line1.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine2 = line2.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine3 = line3.Split('\t').Select(str => float.Parse(str)).ToArray();
        var numbersLine4 = line4.Split('\t').Select(str => float.Parse(str)).ToArray();
#endif

        var mat = new Matrix4x4();

        mat[0] = numbersLine1[0];
        mat[1] = numbersLine1[1];
        mat[2] = numbersLine1[2];
        mat[3] = numbersLine1[3];

        mat[4] = numbersLine2[0];
        mat[5] = numbersLine2[1];
        mat[6] = numbersLine2[2];
        mat[7] = numbersLine2[3];

        mat[8] = numbersLine3[0];
        mat[9] = numbersLine3[1];
        mat[10] = numbersLine3[2];
        mat[11] = numbersLine3[3];

        mat[12] = numbersLine4[0];
        mat[13] = numbersLine4[1];
        mat[14] = numbersLine4[2];
        mat[15] = numbersLine4[3];

        return mat.transpose;
    }

    public Matrix4x4 ChangeReferenceOfMatrix(Matrix4x4 m)
    {
        var a = YAxisInversion;
        var b = a.inverse;
        return b * m * a;
    }

    private void OnDestroy()
    {
#if !USE_RECORDED_DATA
        SlamWrapper.shutdown_slam_system(_slamSystem);
#endif
    }

    void Update()
    {
        Matrix4x4 m2 = Matrix4x4.identity;
        bool isM2Set = false;
        if (!_bypassUpdate)
        {
            _texture2D.LoadImage(File.ReadAllBytes(_images[current_frame]));
            _texture2D.Apply();

#if !USE_RECORDED_DATA
            var textureBuffer = _texture2D.GetRawTextureData();
            for (int i = 0; i < _buffer_length; i++)
            {
                if (i % 4 != 3)
                {
                    var offset = i % 4;
                    var new_index = (i / 4) * 3 + offset;
                    var j = _new_buffer_length - 1 - new_index;
                    _fixed_buffer2[j] = textureBuffer[i];
                }
            }

            var index = 0;

            for (int j = 0; j < _height; j++)
            {
                for (int i = 0; i < _width; i++)
                {
                    var swap = j * _width * 3 + (_width - 1 - i) * 3;
                    for (int channel = 0; channel < 3; channel++)
                    {
                        var new_index = swap + channel;

                        if (new_index < 0 || new_index >= _fixed_buffer2.Length)
                        {
                            Debug.LogError("new_index=" + new_index + " i=" + i + " j=" + j + " channel=" + channel);
                        }

                        _fixed_buffer[index] = _fixed_buffer2[new_index];
                        index++;
                    }
                }
            }

            IntPtr pose = SlamWrapper.update_image(
                            _slamSystem,
                            _pointer,
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
                    m2[i] = _matrix[i];
                }

                isM2Set = true;
                m2 = m2.transpose;

                File.AppendAllText(@"C:\Users\sesa455926\Desktop\movieSlam2\poses.txt",
                        "" + current_frame + " pose is \n[" + m2.ToString() + "]\n"
                    );

                if (current_frame == 665)
                {
                    int array_size = 0;
                    var array_ptr =
                        SlamWrapper.get_3d_tracked_points(_slamSystem, ref array_size);

                    if (array_ptr.ToInt64() != 0 && array_size > 0)
                    {
                        var tracked3DPoints = new float[array_size];
                        Marshal.Copy(array_ptr, tracked3DPoints, 0, array_size);
                        SlamWrapper.delete_pointer(array_ptr);

                        var points = new List<Vector3>();
                        for (int i = 0; i < tracked3DPoints.Length / 3; i++)
                        {
                            var x = tracked3DPoints[i * 3 + 0];
                            var y = tracked3DPoints[i * 3 + 1];
                            var z = tracked3DPoints[i * 3 + 2];
                            points.Add(new Vector3(x, y, z));
                        }

                        var lines =
                            points
                            .Select((point) => "" + point.x + " " + point.y + " " + point.z)
                            .ToArray();

                        var path = @"C:\Users\sesa455926\Desktop\movieSlam2\mappoints.txt";
                        File.WriteAllLines(path, lines);
                    }
                }
            }
#else

#if USE_MACHINE_HALL
            var current_index = dict.Keys.ToArray()[current_frame];
#else
            var current_index = current_frame;
#endif
            if (dict.ContainsKey(current_index))
            {
                m2 = dict[current_index];
                isM2Set = true;
            }
#endif
            if (isM2Set)
            {
                var matUnity = ToUnityFrame(m2);
                //ApplyPoseMatrix(matUnity);
            }

            Matrix4x4 mat;
            var hasDetected = ArucoDetection(_texture2D, out mat);
            if (hasDetected)
            {
                ApplyPoseMatrix(mat);

                if (isM2Set)
                {
                    var matUnity = ToUnityFrame(m2);
                    if (!_pairOfMatrices.ContainsKey(current_frame))
                    {
                        _pairOfMatrices.Add(
                            current_frame,
                            new PairOfMatrixAtFrame()
                            {
                                matrix_slam = matUnity,
                                matrix_aruco = mat,
                            });
                    }
                }
            }
        }
        
        
        if (current_frame == 664)
        {
            List<float> ratios = new List<float>();
            for (int i = 0; i < 644; i++)
            {
                if 
                (
                    _pairOfMatrices.ContainsKey(i) && 
                    _pairOfMatrices.ContainsKey(i+1)
                )
                {
                    var matrices_i = _pairOfMatrices[i];
                    var matrices_i_plus_one = _pairOfMatrices[i + 1];

                    var aruco_pos_i = 
                        new Vector3
                        (
                            matrices_i.matrix_aruco.m03, 
                            matrices_i.matrix_aruco.m13, 
                            matrices_i.matrix_aruco.m23
                        );

                    var aruco_pos_i_one =
                        new Vector3
                        (
                            matrices_i_plus_one.matrix_aruco.m03,
                            matrices_i_plus_one.matrix_aruco.m13,
                            matrices_i_plus_one.matrix_aruco.m23
                        );
                    
                    var slam_pos_i =
                        new Vector3
                        (
                            matrices_i.matrix_slam.m03,
                            matrices_i.matrix_slam.m13,
                            matrices_i.matrix_slam.m23
                        );

                    var slam_pos_i_one =
                        new Vector3
                        (
                            matrices_i_plus_one.matrix_slam.m03,
                            matrices_i_plus_one.matrix_slam.m13,
                            matrices_i_plus_one.matrix_slam.m23
                        );

                    var delta_slam = (slam_pos_i_one - slam_pos_i).magnitude;
                    var delta_aruco = (aruco_pos_i_one - aruco_pos_i).magnitude;

                    float current_ratio = delta_slam / delta_aruco;

                    if (current_ratio != 0 && !(float.IsInfinity(current_ratio)) && !(float.IsNaN(current_ratio)))
                    {
                        ratios.Add(current_ratio);
                    }
                }
            }

            foreach(var ratio in ratios)
            {
                Debug.Log("ratio is " + ratio);
            }

            float average = ratios.Average();

            Debug.LogWarning("average is " + average);
        }

        if (Time.frameCount %  1 == 0)
        {
            current_frame++;
#if !USE_RECORDED_DATA
            current_frame = current_frame % _images.Count;
#else
#if USE_MACHINE_HALL
            current_frame = current_frame % dict.Keys.Count;
#else
            current_frame = current_frame % (int) dict.Keys.Max();
            if (current_frame == 0)
            {
                current_frame = 350;
            }
#endif
#endif
        }
    }

    private bool ArucoDetection(Texture2D texture2D, out Matrix4x4 mat)
    {
        Mat rgbMat = new Mat((int) _height, (int) _width, CvType.CV_8UC3);
        Utils.texture2DToMat(texture2D, rgbMat);
        Aruco.detectMarkers(rgbMat, _dictionary, _corners, _ids, _detectorParams, _rejected);

        if (_ids.total() > 0)
        {
            Aruco.estimatePoseSingleMarkers(_corners, _markerLength, _camMatrix, _distCoeffs, _rvecs, _tvecs);
            var mat2 = GetMatrix(_tvecs.get(0,0), _rvecs.get(0,0));
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

    private Matrix4x4 ToUnityFrame(Matrix4x4 m2)
    {
        var m3 = ChangeReferenceOfMatrix(m2);
        var m = m3.inverse;
        return m;
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
}
