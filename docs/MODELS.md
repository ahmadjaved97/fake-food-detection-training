# Model Mappings

The configs use likely current timm pretrained names. Verify names in your installed timm version:

```bash
python scripts/list_timm_models.py "resnet*" "deit*" "vit*_patch16_224*" "*dinov2*" "mobilenetv3*" "mobilenetv4*" "swin*" "convnext*"
```

Mappings:

- ResNet-18: `resnet18.tv_in1k`
- ResNet-34: `resnet34.tv_in1k`
- ResNet-50: `resnet50.tv_in1k`
- ResNet-101: `resnet101.tv_in1k`
- DeiT tiny: `deit_tiny_patch16_224.fb_in1k`
- DeiT small: `deit_small_patch16_224.fb_in1k`
- DeiT base: `deit_base_patch16_224.fb_in1k`
- DeiT large fallback: `deit3_large_patch16_224.fb_in22k_ft_in1k`
- ViT tiny: `vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- ViT small: `vit_small_patch16_224.augreg_in21k_ft_in1k`
- ViT base: `vit_base_patch16_224.augreg_in21k_ft_in1k`
- ViT large: `vit_large_patch16_224.augreg_in21k_ft_in1k`
- DINOv2 tiny fallback: `vit_small_patch14_dinov2.lvd142m`
- DINOv2 small: `vit_small_patch14_dinov2.lvd142m`
- DINOv2 base: `vit_base_patch14_dinov2.lvd142m`
- DINOv2 large: `vit_large_patch14_dinov2.lvd142m`

DINOv2 configs set `model.freeze_backbone: true` by default, so only the classifier head is trained unless you override it.
- MobileNetV3 small 100: `mobilenetv3_small_100.lamb_in1k`
- MobileNetV3 large 075: `mobilenetv3_large_075` with `pretrained: false`
- MobileNetV3 large 100: `mobilenetv3_large_100.ra_in1k`
- MobileNetV4 conv small: `mobilenetv4_conv_small.e2400_r224_in1k`
- MobileNetV4 conv medium: `mobilenetv4_conv_medium.e500_r256_in1k`
- MobileNetV4 conv large: `mobilenetv4_conv_large.e600_r384_in1k`
- MobileNetV4 hybrid medium: `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k`
- MobileNetV4 hybrid large: `mobilenetv4_hybrid_large.e600_r384_in1k`
- Swin tiny: `swin_tiny_patch4_window7_224.ms_in1k`
- Swin small: `swin_small_patch4_window7_224.ms_in1k`
- Swin base: `swin_base_patch4_window7_224.ms_in1k`
- Swin base 22k fine-tuned: `swin_base_patch4_window7_224.ms_in22k_ft_in1k`
- Swin large 22k fine-tuned: `swin_large_patch4_window7_224.ms_in22k_ft_in1k`
- ConvNeXt tiny: `convnext_tiny.fb_in22k_ft_in1k`
- ConvNeXt small: `convnext_small.fb_in22k_ft_in1k`
- ConvNeXt base: `convnext_base.fb_in22k_ft_in1k`

If a name is unavailable, update `model.name` in the YAML to a valid value returned by `timm.list_models(pattern, pretrained=True)`.
