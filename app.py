from algorithms import Index, process_image


def run() -> None:
    filename = '/Users/mcosgriff/Downloads/18FEB23084801-A3DS-057798936010_cal_2016v0_scube_native.tif'

    process_image(Index.WORLD_VIEW_WATER_INDEX, filename)


if __name__ == '__main__':
    run()
