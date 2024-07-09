from sentence_transformers import CrossEncoder
 
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)
# scores = model.predict([('How many people live in Berlin?', 'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'), 
#                         ('How many people live in Berlin?', 'Berlin is well known for its museums.')])

scores = model.predict([('人群', '小明的生日宴会上，一群人站在桌子上，他们正在吃蛋糕和喝饮料。'),
                        ('人', '小明的生日宴会上，一群人站在桌子上，他们正在吃蛋糕和喝饮料。'), 
                        ('小明', '小明的生日宴会上，一群人站在桌子上，他们正在吃蛋糕和喝饮料。'), 
                        ('小花', '小明的生日宴会上，一群人站在桌子上，他们正在吃蛋糕和喝饮料。'), 
                        ('人', '熊猫喜欢吃辣椒')])
print(scores)