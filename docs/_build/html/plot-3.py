import plot
height = 200
width = 200
srces =  [[43.71, 22.91, 10.54, 3.34],
              [101.62, 40.60, 1.37, 3.40],
              [92.63, 110.56, 1.81, 3.66],
              [183.60, 85.90, 1.23, 5.06],
              [34.12, 162.54, 1.95, 6.02],
              [153.87, 169.18, 1.06, 6.61],
              [155.54, 32.14, 1.46, 4.05],
              [130.56, 183.48, 1.63, 4.11]] 
im_data = plot.make_source(srces, height, width)