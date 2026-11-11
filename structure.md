.
│
├── models
│   └── Hunyuan3D-2
│       ├── hunyuan3d-delight-v2-0
│       │   ├── feature_extractor
│       │   │   └── preprocessor_config.json
│       │   ├── model_index.json
│       │   ├── scheduler
│       │   │   └── scheduler_config.json
│       │   ├── text_encoder
│       │   │   ├── config.json
│       │   │   └── model.safetensors
│       │   ├── tokenizer
│       │   │   ├── merges.txt
│       │   │   ├── special_tokens_map.json
│       │   │   ├── tokenizer_config.json
│       │   │   └── vocab.json
│       │   ├── unet
│       │   │   ├── config.json
│       │   │   └── diffusion_pytorch_model.safetensors
│       │   └── vae
│       │       ├── config.json
│       │       └── diffusion_pytorch_model.safetensors
│       ├── hunyuan3d-dit-v2-0-turbo
│       │   ├── config.yaml
│       │   ├── model.fp16.ckpt
│       │   └── model.fp16.safetensors
│       ├── hunyuan3d-dit-v2-mv-turbo
│       │   ├── config.yaml
│       │   ├── model.fp16.ckpt
│       │   ├── model.fp16.safetensors
│       │   └── README.md
│       ├── hunyuan3d-paint-v2-0-turbo
│       │   ├── feature_extractor
│       │   │   └── preprocessor_config.json
│       │   ├── image_encoder
│       │   │   ├── config.json
│       │   │   ├── model.safetensors
│       │   │   └── preprocessor_config.json
│       │   ├── model_index.json
│       │   ├── README.md
│       │   ├── scheduler
│       │   │   └── scheduler_config.json
│       │   ├── text_encoder
│       │   │   ├── config.json
│       │   │   └── pytorch_model.bin
│       │   ├── tokenizer
│       │   │   ├── merges.txt
│       │   │   ├── special_tokens_map.json
│       │   │   ├── tokenizer_config.json
│       │   │   └── vocab.json
│       │   ├── unet
│       │   │   ├── config.json
│       │   │   ├── diffusion_pytorch_model.bin
│       │   │   ├── diffusion_pytorch_model.safetensors
│       │   │   └── modules.py
│       │   └── vae
│       │       ├── config.json
│       │       ├── diffusion_pytorch_model.bin
│       │       └── diffusion_pytorch_model.safetensors
│       ├── hunyuan3d-vae-v2-0-turbo
│       │   ├── config.yaml
│       │   ├── model.fp16.ckpt
│       │   └── model.fp16.safetensors
│       └── HunyuanWorld-Mirror
│           ├── app.py
│           ├── arch.png
│           ├── config.json
│           ├── configuration.json
│           ├── License.txt
│           ├── model.safetensors
│           ├── Notice.txt
│           ├── qrcode
│           │   ├── discord.png
│           │   ├── wechat.png
│           │   ├── xiaohongshu.png
│           │   └── x.png
│           ├── README.md
│           ├── requirements.txt
│           ├── skyseg.onnx
│           ├── src
│           │   ├── __init__.py
│           │   ├── models
│           │   │   ├── heads
│           │   │   │   ├── camera_head.py
│           │   │   │   ├── dense_head.py
│           │   │   │   └── __pycache__
│           │   │   │       ├── camera_head.cpython-312.pyc
│           │   │   │       └── dense_head.cpython-312.pyc
│           │   │   ├── __init__.py
│           │   │   ├── layers
│           │   │   │   ├── attention.py
│           │   │   │   ├── block.py
│           │   │   │   ├── drop_path.py
│           │   │   │   ├── __init__.py
│           │   │   │   ├── layer_scale.py
│           │   │   │   ├── mlp.py
│           │   │   │   ├── patch_embed.py
│           │   │   │   ├── __pycache__
│           │   │   │   │   ├── attention.cpython-312.pyc
│           │   │   │   │   ├── block.cpython-312.pyc
│           │   │   │   │   ├── drop_path.cpython-312.pyc
│           │   │   │   │   ├── __init__.cpython-312.pyc
│           │   │   │   │   ├── layer_scale.cpython-312.pyc
│           │   │   │   │   ├── mlp.cpython-312.pyc
│           │   │   │   │   ├── patch_embed.cpython-312.pyc
│           │   │   │   │   ├── rope.cpython-312.pyc
│           │   │   │   │   ├── swiglu_ffn.cpython-312.pyc
│           │   │   │   │   └── vision_transformer.cpython-312.pyc
│           │   │   │   ├── rope.py
│           │   │   │   ├── swiglu_ffn.py
│           │   │   │   └── vision_transformer.py
│           │   │   ├── models
│           │   │   │   ├── __pycache__
│           │   │   │   │   ├── rasterization.cpython-312.pyc
│           │   │   │   │   ├── visual_transformer.cpython-312.pyc
│           │   │   │   │   └── worldmirror.cpython-312.pyc
│           │   │   │   ├── rasterization.py
│           │   │   │   ├── visual_transformer.py
│           │   │   │   └── worldmirror.py
│           │   │   ├── __pycache__
│           │   │   │   └── __init__.cpython-312.pyc
│           │   │   └── utils
│           │   │       ├── act_gs.py
│           │   │       ├── camera_utils.py
│           │   │       ├── frustum.py
│           │   │       ├── geometry.py
│           │   │       ├── grid.py
│           │   │       ├── priors.py
│           │   │       ├── __pycache__
│           │   │       │   ├── act_gs.cpython-312.pyc
│           │   │       │   ├── camera_utils.cpython-312.pyc
│           │   │       │   ├── frustum.cpython-312.pyc
│           │   │       │   ├── geometry.cpython-312.pyc
│           │   │       │   ├── grid.cpython-312.pyc
│           │   │       │   ├── priors.cpython-312.pyc
│           │   │       │   ├── rotation.cpython-312.pyc
│           │   │       │   └── sh_utils.cpython-312.pyc
│           │   │       ├── rotation.py
│           │   │       └── sh_utils.py
│           │   ├── __pycache__
│           │   │   └── __init__.cpython-312.pyc
│           │   └── utils
│           │       ├── build_pycolmap_recon.py
│           │       ├── color_map.py
│           │       ├── cropping.py
│           │       ├── geometry.py
│           │       ├── gs_effects.py
│           │       ├── inference_utils.py
│           │       ├── __init__.py
│           │       ├── __pycache__
│           │       │   ├── color_map.cpython-312.pyc
│           │       │   ├── geometry.cpython-312.pyc
│           │       │   ├── gs_effects.cpython-312.pyc
│           │       │   ├── inference_utils.cpython-312.pyc
│           │       │   ├── __init__.cpython-312.pyc
│           │       │   ├── render_utils.cpython-312.pyc
│           │       │   ├── save_utils.cpython-312.pyc
│           │       │   ├── video_utils.cpython-312.pyc
│           │       │   ├── visual_util.cpython-312.pyc
│           │       │   └── warnings.cpython-312.pyc
│           │       ├── render_utils.py
│           │       ├── save_utils.py
│           │       ├── video_utils.py
│           │       ├── visual_util.py
│           │       └── warnings.py
│           └── teaser.jpg
├── server
│   ├── api
│   │   ├── api_server.py
│   │   ├── enhanced_api_server.py
│   │   └── volcano_engine_client.py
│   ├── client
│   │   ├── app.py
│   │   ├── static
│   │   │   ├── icon_no_bg.ico
│   │   │   ├── icon_no_bg.png
│   │   │   └── img
│   │   │       ├── example0.png
│   │   │       ├── example1.png
│   │   │       └── example2.png
│   │   └── templates
│   │       ├── AR.html
│   │       ├── gen.html
│   │       ├── index.html
│   │       ├── reveal.html
│   │       └── space.html
│   ├── database
│   │   ├── models.db
│   │   └── sqlite.py
│   ├── README.md
│   ├── static
│   │   ├── icon_no_bg.ico
│   │   ├── icon_no_bg.png
│   │   └── img
│   │       ├── example0.png
│   │       ├── example1.png
│   │       └── example2.png
│   ├── temp
│   │   ├── 05b46207-7797-40af-9618-a9305203b907_white.glb
│   │   ├── 0bd0755f-76f5-4b8f-9786-d7401f435b39_picture.png
│   │   ├── 0bd0755f-76f5-4b8f-9786-d7401f435b39_white.glb
│   │   ├── 0cf24ea1-6227-4386-a36d-7dd4fa3d896b_white.glb
│   │   ├── 1ea05dda-eae2-445b-be94-a9e646b0b4eb_white.glb
│   │   ├── 2098e9a9-36fd-4553-bf1c-f65e5f350235_picture.png
│   │   ├── 2098e9a9-36fd-4553-bf1c-f65e5f350235_white.glb
│   │   ├── 2515c15f-0e65-421e-afdf-d8555b46be45_white.glb
│   │   ├── 313f0e14-691e-4db0-98eb-25f30f819be9_white.glb
│   │   ├── 3321e79f-6b89-4553-a01f-0ea9cdbf7571_white.glb
│   │   ├── 379e0dd5-f045-4ea3-9dcf-0181202f7ce5_white.glb
│   │   ├── 4f52c7cd-0e83-47bd-8561-aa9f9964599f_picture.png
│   │   ├── 4f52c7cd-0e83-47bd-8561-aa9f9964599f_white.glb
│   │   ├── 54d99b5b-985f-4187-90cd-e94b542881ff_white.glb
│   │   ├── 55a8c033-b74c-42db-afe8-b5f36f255b9b_white.glb
│   │   ├── 76d9ffcf-79e0-4d0d-8a02-5b657dc4af24_white.glb
│   │   ├── 9c15a574-247d-45a4-80b1-e794853795ae_white.glb
│   │   ├── ab7744c6-1f64-4a8c-9243-a868ca5b286d_white.glb
│   │   ├── af19655c-85ec-41f4-8269-be409de7ee6a_white.glb
│   │   ├── beb79df8-bd3e-45b4-b44c-7d16bd11fe7a_picture.png
│   │   ├── e1c61cb0-b063-446e-8c37-73c548db7918_picture.png
│   │   └── e1c61cb0-b063-446e-8c37-73c548db7918_white.glb
│   └── uploads
│       ├── 05b46207-7797-40af-9618-a9305203b907_textured.glb
│       ├── 0bd0755f-76f5-4b8f-9786-d7401f435b39_textured.glb
│       ├── 0cf24ea1-6227-4386-a36d-7dd4fa3d896b_textured.glb
│       ├── 12d63afb-ae11-4599-a427-b31262e64dcd_textured.glb
│       ├── 1ea05dda-eae2-445b-be94-a9e646b0b4eb_textured.glb
│       ├── 2098e9a9-36fd-4553-bf1c-f65e5f350235_textured.glb
│       ├── 2515c15f-0e65-421e-afdf-d8555b46be45_textured.glb
│       ├── 313f0e14-691e-4db0-98eb-25f30f819be9_textured.glb
│       ├── 3321e79f-6b89-4553-a01f-0ea9cdbf7571_textured.glb
│       ├── 379e0dd5-f045-4ea3-9dcf-0181202f7ce5_textured.glb
│       ├── 391bec71-44b7-4739-bbca-315ac720f11d_scene.glb
│       ├── 43f2bfae-5676-4e2e-80c5-db0c03a419ec_scene.glb
│       ├── 47b46e2b-e758-4022-8c3b-9fd5804e09ae_scene.glb
│       ├── 4f52c7cd-0e83-47bd-8561-aa9f9964599f_textured.glb
│       ├── 54d99b5b-985f-4187-90cd-e94b542881ff_textured.glb
│       ├── 55a8c033-b74c-42db-afe8-b5f36f255b9b_textured.glb
│       ├── 76d9ffcf-79e0-4d0d-8a02-5b657dc4af24_textured.glb
│       ├── 8054ac55-76cd-4051-adde-3937ea8fc4ca_textured.glb
│       ├── 8717ceed-a84c-4205-b0f2-71e5c0567ab8_scene.glb
│       ├── 9c15a574-247d-45a4-80b1-e794853795ae_textured.glb
│       ├── ab7744c6-1f64-4a8c-9243-a868ca5b286d_textured.glb
│       ├── af19655c-85ec-41f4-8269-be409de7ee6a_textured.glb
│       └── e1c61cb0-b063-446e-8c37-73c548db7918_textured.glb
└── structure.md