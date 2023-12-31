from src.nlps.approach.transformer import TransformerModelArgument
from src.nlps.argument import argument_class, ArgumentPool


Name2Backbone = {
    "bart": "facebook/bart-base",
    "t5": "t5-base"
}


@argument_class
class PGSTModelArgument(TransformerModelArgument):
    def __post_init__(self):
        super().__post_init__()
        name, backbone = self.plm_name_or_path.split("-")
        assert name == ArgumentPool.meta_arguments.approach
        assert backbone in Name2Backbone
        self.backbone_name = backbone

    @property
    def backbone(self):
        return Name2Backbone[self.backbone_name]


PTRModelArgument = PGSTModelArgument
