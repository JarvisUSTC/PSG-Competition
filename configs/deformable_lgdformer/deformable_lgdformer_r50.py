model = dict(
    type='PSGTr',
    backbone=dict(type='ResNet',
                  depth=50,
                  num_stages=4,
                  out_indices=(0, 1, 2, 3),
                  frozen_stages=1,
                  norm_cfg=dict(type='BN', requires_grad=False),
                  norm_eval=True,
                  style='pytorch',
                  init_cfg=dict(type='Pretrained',
                                checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='LGDFormerHead',
        num_classes=80,
        num_relations=117,
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(type='DetrTransformerEncoder',
                         num_layers=6,
                         transformerlayers=dict(
                             type='BaseTransformerLayer',
                             attn_cfgs=[
                                 dict(type='MultiheadAttention',
                                      embed_dims=256,
                                      num_heads=8,
                                      dropout=0.1)
                             ],
                             feedforward_channels=2048,
                             ffn_dropout=0.1,
                             operation_order=('self_attn', 'norm', 'ffn',
                                              'norm'))),
            decoder=dict(type='DetrTransformerDecoder',
                          return_intermediate=True,
                          num_layers=6,
                          transformerlayers=dict(
                              type='DetrTransformerDecoderLayer',
                              attn_cfgs=dict(type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1),
                              feedforward_channels=2048,
                              ffn_dropout=0.1,
                              operation_order=('self_attn', 'norm',
                                               'cross_attn', 'norm', 'ffn',
                                               'norm'))),
        ),
        panoptic_head=dict(
            type='DeformableDETRMaskHead',
            num_query=300,
            num_things_classes=80,
            num_stuff_classes=53,
            num_classes=133,
            in_channels=2048,
            sync_cls_avg_factor=True,
            as_two_stage=False,
            mask_assigner=dict(
                type='BoxMaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            transformer=dict(
                type='DeformableDetrTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    use_checkpoint=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='DETRBaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    use_checkpoint=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm')))),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0)),
        predicate_node_generator=dict(
            type='Predicate_Node_Generator',
            num_classes=80,
            rel_encoder=dict(type='DetrTransformerEncoder',
                         num_layers=3,
                         transformerlayers=dict(
                             type='BaseTransformerLayer',
                             attn_cfgs=[
                                 dict(type='MultiheadAttention',
                                      embed_dims=256,
                                      num_heads=8,
                                      dropout=0.1)
                             ],
                             feedforward_channels=2048,
                             ffn_dropout=0.1,
                             operation_order=('self_attn', 'norm', 'ffn',
                                              'norm'))),
            rel_decoder=dict(type='DetrTransformerDecoder',
                          return_intermediate=True,
                          num_layers=6,
                          transformerlayers=dict(
                              type='DetrTransformerDecoderLayer',
                              attn_cfgs=dict(type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1),
                              feedforward_channels=2048,
                              ffn_dropout=0.1,
                              operation_order=('self_attn', 'norm',
                                               'cross_attn', 'norm', 'ffn',
                                               'norm'))),
            rel_q_gen_decoder=dict(type='DetrTransformerDecoder',
                          return_intermediate=True,
                          num_layers=1,
                          transformerlayers=dict(
                              type='DetrTransformerDecoderLayer',
                              attn_cfgs=dict(type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1),
                              feedforward_channels=2048,
                              ffn_dropout=0.1,
                              operation_order=('self_attn', 'norm',
                                               'cross_attn', 'norm', 'ffn',
                                               'norm'))),
            entities_aware_decoder=dict(type='DetrTransformerDecoder',
                          return_intermediate=True,
                          num_layers=6,
                          transformerlayers=dict(
                              type='DetrTransformerDecoderLayer',
                              attn_cfgs=dict(type='MultiheadAttention',
                                             embed_dims=256,
                                             num_heads=8,
                                             dropout=0.1),
                              feedforward_channels=2048,
                              ffn_dropout=0.1,
                              operation_order=('self_attn', 'norm',
                                               'cross_attn', 'norm', 'ffn',
                                               'norm'))),                            
        ),
        positional_encoding=dict(type='SinePositionalEncoding',
                                 num_feats=128,
                                 normalize=True),
        sub_loss_cls=dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0,
                            class_weight=1.0),
        sub_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        sub_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        sub_focal_loss=dict(type='BCEFocalLoss', loss_weight=2.0),
        sub_dice_loss=dict(type='psgtrDiceLoss', loss_weight=2.0),
        obj_loss_cls=dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=1.0,
                            class_weight=1.0),
        obj_loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        obj_loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        obj_focal_loss=dict(type='BCEFocalLoss', loss_weight=2.0),
        obj_dice_loss=dict(type='psgtrDiceLoss', loss_weight=2.0),
        rel_loss_cls=dict(type='CrossEntropyLoss',
                          use_sigmoid=False,
                          loss_weight=2.0,
                          class_weight=1.0),
        sub_id_loss=dict(type='MultilabelCrossEntropy', loss_weight=2.0),
        obj_id_loss=dict(type='MultilabelCrossEntropy', loss_weight=2.0),
        loss_cls=dict(type='CrossEntropyLoss',
                      use_sigmoid=False,
                      loss_weight=4.0,
                      class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=3.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        focal_loss=dict(type='BCEFocalLoss', loss_weight=1.0),
        dice_loss=dict(type='psgtrDiceLoss', loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(id_assigner=dict(type='IdMatcher',
                                    sub_id_cost=dict(type='ClassificationCost',
                                                     weight=1.),
                                    obj_id_cost=dict(type='ClassificationCost',
                                                     weight=1.),
                                    r_cls_cost=dict(type='ClassificationCost',
                                                    weight=1.)),
                   bbox_assigner=dict(type='HungarianAssigner',
                                      cls_cost=dict(type='ClassificationCost',
                                                    weight=4.0),
                                      reg_cost=dict(type='BBoxL1Cost',
                                                    weight=3.0),
                                      iou_cost=dict(type='IoUCost',
                                                    iou_mode='giou',
                                                    weight=2.0))),
    test_cfg=dict(max_per_img=100,
                logit_adj_tau=0.0))