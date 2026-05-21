from collections.abc import Sequence

import torch
import torch.nn.functional as F
import numpy as np
from metaworld_dataset.domain import Attribute, Text, Action



class NormalizeAttributes:
    '''
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_size = self.max_size - self.min_size

        self.image_size = image_size
        self.min_position = self.max_size // 2
        self.max_position = self.image_size - self.min_position
        self.scale_position = self.max_position - self.min_position
    '''
    def __call__(self, attr: Attribute) -> Attribute:
        #Normalizes attributes between -1 and 1
        wall = attr.wall
        wall[0] /= 0.05
        wall[1] = np.clip(wall[1],-1,1)
        
        soccer_goal = attr.soccer_goal
        soccer_goal[0] /= 0.1
        soccer_goal[1] = np.clip(soccer_goal[1],-1,1)

        ball = 2*((attr.ball+11)/(1+11)) - 1
        ball = np.clip(ball,-1,1)
        
        return Attribute(
            proprio_x = attr.proprio_x/0.525,
            proprio_y= 2*((attr.proprio_y-0.348) / (1.025-0.348)) - 1,
            proprio_z=2*((attr.proprio_z+0.525) / (0.7+0.525)) - 1,
            proprio_gripper=attr.proprio_gripper,
            ball = ball,
            wall = wall,
            soccer_goal = soccer_goal,
            unpaired=attr.unpaired,
        )

class NormalizeActions:
    '''
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_size = self.max_size - self.min_size

        self.image_size = image_size
        self.min_position = self.max_size // 2
        self.max_position = self.image_size - self.min_position
        self.scale_position = self.max_position - self.min_position
    '''
    def __call__(self, attr: Attribute) -> Attribute:
        return Action(
            dis_x = np.clip(attr.dis_x,-1,1),
            dis_y = np.clip(attr.dis_y,-1,1),
            dis_z = np.clip(attr.dis_z,-1,1),
            gripper = np.clip(attr.gripper,-1,1),
            unpaired=attr.unpaired,
        )
    
def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


class UnnormalizeAttributes:
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_size = self.max_size - self.min_size

        self.image_size = image_size
        self.min_position = self.max_size // 2
        self.max_position = self.image_size - self.min_position
        self.scale_position = self.max_position - self.min_position

    def __call__(self, attr: Attribute) -> Attribute:
        return Attribute(
            category=attr.category,
            x=to_unit_range(attr.x) * self.scale_position + self.min_position,
            y=to_unit_range(attr.y) * self.scale_position + self.min_position,
            size=to_unit_range(attr.size) * self.scale_size + self.min_size,
            rotation=attr.rotation,
            color_r=to_unit_range(attr.color_r) * 255,
            color_g=to_unit_range(attr.color_g) * 255,
            color_b=to_unit_range(attr.color_b) * 255,
            unpaired=attr.unpaired,
        )


def attribute_to_tensor(attr) -> list[torch.Tensor]:
    tensors = [t.reshape(-1) for t in attr if isinstance(t, torch.Tensor)]
    vec = torch.cat(tensors)
    #print(vec.min(),vec.max())
    tensors = vec
    if attr.unpaired is not None:
        tensors.append(attr.unpaired)
    return tensors


def color_blind_visual_domain(image: torch.Tensor) -> torch.Tensor:
    return image.mean(dim=0, keepdim=True).expand(3, -1, -1)


def text_to_bert(text: Text) -> torch.Tensor:
    return text.bert


class TextAndAttrs:
    def __init__(self, min_size: int = 7, max_size: int = 14, image_size: int = 32):
        self.normalize = NormalizeAttributes(min_size, max_size, image_size)

    def __call__(self, x: Text) -> dict[str, torch.Tensor]:
        text: dict[str, torch.Tensor] = {"bert": x.bert}
        attr = self.normalize(x.attr)
        attr_list = attribute_to_tensor(attr)
        text["cls"] = attr_list[0]
        text["attr"] = attr_list[1]
        if len(attr_list) == 3:
            text["unpaired"] = attr_list[2]
        grammar_categories = structure_category_from_choice(composer, x.choice)
        text.update(
            {
                name: torch.Tensor([category])
                for name, category in grammar_categories.items()
            }
        )
        return text


def attr_to_str(
    attr: Attribute, grammar_predictions: dict[str, list[int]]
) -> list[str]:
    captions: list[str] = []
    choices = choices_from_structure_categories(composer, grammar_predictions)
    for k in range(attr.category.size(0)):
        caption, _ = composer(
            {
                "shape": attr.category[k].detach().cpu().item(),
                "rotation": attr.rotation[k].detach().cpu().item(),
                "color": (
                    attr.color_r[k].detach().cpu().item(),
                    attr.color_g[k].detach().cpu().item(),
                    attr.color_b[k].detach().cpu().item(),
                ),
                "size": attr.size[k].detach().cpu().item(),
                "location": (
                    attr.x[k].detach().cpu().item(),
                    attr.y[k].detach().cpu().item(),
                ),
            },
            choices[k],
        )
        captions.append(caption)
    return captions
