import numpy as np
import torch

import random
class GenAugPrompt():
    def room_name(self):
        rooms = [ 'restaurant', 'living room', 'lounge', 'kitchen', 'dinning room',
                  'lab', 'classroom', 'sunroom', 'office', 'robotics lab'
                 ]
        room = random.choice(rooms)
        output = 'a picture taken above a table of views inside a {0}'.format(room)
        return output

    def material_name(self):

        materials = ['straw', 'metal', 'wooden', 'silky', 'marble', 'fabric', 'shiny', 'transparent', 'wool',
                     'plastic', 'glass', 'ceramic', 'bamboo', 'paper', 'aluminum', 'iron', 'copper',
                     'gold', 'bronze', 'crystal', 'creamy', 'leather',
                     'white', 'orange', 'yellow', 'blue', 'pink', 'red',
                     'purple', 'black', 'brown', 'gray', 'green', 'magenta', 'olive', 'navy', 'colorful'
                     ]


        return random.choice(materials)

    def table_name(self):
        materials = ['wooden',
                     'marble',
                     'plastic',
                     'white',
                     'red',
                     'white',
                     'orange',
                     'yellow', 'blue', 'pink', 'red',
                     'brown', 'green', 'olive', 'navy',
                     'black', 'gray', 'colorful'
                     ]

        others = ['3d render', 'photorealisitc', '8k', 'natural light', 'cinematic lighting', 'detailed', 'bright',
                  'contemporary', 'amazing', 'highly detailed', 'magic']
        tables = ['dining table', 'table', 'coffee table', 'desk', 'office table']
        material = random.choice(materials)
        other = random.choice(others)
        table = random.choice(tables)
        table_prompt = "a top down view of a large and clean{0} {1} surface, {2}".format(material, table, other)
        return table_prompt
