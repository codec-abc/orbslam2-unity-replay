Shader "Custom/undistordShader" 
{
	Properties
	{
		_MainTex("MainTexture", 2D) = "white" {}

		_Fx("Fx", Float) = 600
		_Fy("Fy", Float) = 600
		_Cx("Cx", Float) = 320
		_Cy("Cy", Float) = 240

		_K1("K1", Float) = 0
		_K2("K2", Float) = 0
		_P1("P1", Float) = 0
		_P2("P2", Float) = 0
		_K3("K3", Float) = 0

		_Width("Width", Float) = 640.0
		_Height("Height", Float) = 480.0
	}

	SubShader
	{
		Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
		LOD 100

		ZWrite Off
		Blend SrcAlpha OneMinusSrcAlpha

		Pass
		{
			CGPROGRAM

			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;

			float _Fx;
			float _Fy;

			float _Cx;
			float _Cy;

			float _K1;
			float _K2;
			float _P1;
			float _P2;
			float _K3;

			float _Width;
			float _Height;

			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = TRANSFORM_TEX(v.uv, _MainTex);
				return o;
			}

			fixed4 frag(v2f i) : SV_Target
			{
				float cx = _Cx;
				float cy = _Height - _Cy;

				float x = (i.uv.x * _Width - cx) / _Fx;
				float y = (i.uv.y * _Height - cy) / _Fy;

				float x_p = x;
				float y_p = y;

				float x2 = x_p * x_p;
				float y2 = y_p * y_p;
				float r2 = x2 + y2;

				float kp = (1.0f + _K1 * r2 + _K2 * r2*r2 + _K3 * r2*r2*r2);

				float x_pp = 
					x_p * kp  +
					2.0f * _P1 * x_p * y_p + _P2 * (r2 + 2.0f * x_p * x_p);

				float y_pp = 
					y_p * kp +
					2.0f * _P2 * x_p * y_p + _P1 * (r2 + 2.0f * y_p * y_p);

				float map_x = x_pp * _Fx + cx;
				float map_y = y_pp * _Fy + cy;

				float2 new_uv = float2(map_x / _Width, map_y / _Height);

				fixed4 col = tex2D(_MainTex, new_uv);
				col.a = 0.5f;
				return col;
			}
			ENDCG
		}
	}
}