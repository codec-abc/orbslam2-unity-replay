using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class PointCloudDrawer : MonoBehaviour
{
    [SerializeField]
    private string directory_to_png;

    private int current_frame = 0;
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
    private const float sphere_size = 0.02f;

    private static CameraIntrinsics GetIntrinsics()
    {
        return new CameraIntrinsics
        (
            width: 752,
            height: 480,
            fx: 458.654f,
            fy: 457.296f,
            cx: 367.215f,
            cy: 248.375f
        );
    }

    private static DistortionCoefficient GetDistortionCoef()
    {
        return new DistortionCoefficient()
        {
            k1 = -0.28340811f,
            k2 = 0.07395907f,
            p1 = 0.00019359f,
            p2 = 1.76187114e-05f,
            k3 = 0
        };
    }

    Matrix4x4 invertYM = 
        Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));

    private GameObject _root;

    private bool _bypassUpdate = false;

    void Start()
    {
        CreatePointCloud();
        RetrivePoses();
        GetPngs();
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

        var intrinsics = GetIntrinsics();
        var distCoefs = GetDistortionCoef();

        int width = (int) intrinsics.Width;
        int height = (int) intrinsics.Height;

        int halfWidth = width / 2;
        int halfHeight = height / 2;

        _texture2D = new Texture2D(halfWidth, halfHeight, TextureFormat.ARGB32, false);
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
                var line0 = lines[i + 0].Replace("[", "").Replace("]", "").Replace(" ", "");
                var line1 = lines[i + 1].Replace("[", "").Replace("]", "").Replace(" ", "");
                var line2 = lines[i + 2].Replace("[", "").Replace("]", "").Replace(" ", "");
                var line3 = lines[i + 3].Replace("[", "").Replace("]", "").Replace(" ", "");
                
                var mat = GetMatrix(line0, line1, line2, line3);
                dict.Add(current_pose, mat);
                i += 4;
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
        var numbersLine1 = line1.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine2 = line2.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine3 = line3.Split(',').Select(str => float.Parse(str)).ToArray();
        var numbersLine4 = line4.Split(',').Select(str => float.Parse(str)).ToArray();

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
        var a = invertYM;
        var b = a.inverse;
        return b * m * a;
    }
    
    void Update()
    {
        if (!_bypassUpdate)
        {
            var current_index = dict.Keys.ToArray()[current_frame];
            var m2 = dict[current_index];

            var m3 = ChangeReferenceOfMatrix(m2);

            var m = m3.inverse;

            Vector3 t = m.GetColumn(3);

            Quaternion q =
                Quaternion.LookRotation(
                     m.GetColumn(2),
                     m.GetColumn(1)
                );

            _camera.transform.position = t;
            _camera.transform.rotation = q;

            CameraIntrinsics intrinsics = GetIntrinsics();

            var mat = BuildProjMatrixFromIntrinsics(intrinsics, near_plane, far_plane);

            _camera.projectionMatrix = mat;

            _texture2D.LoadImage(File.ReadAllBytes(_images[current_frame]));
            _texture2D.Apply();

            if (Time.frameCount % 3 == 0)
            {
                current_frame++;
                current_frame = current_frame % dict.Keys.Count;
            }

            //Debug.LogWarning("current_frame " + current_frame);
        }
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
