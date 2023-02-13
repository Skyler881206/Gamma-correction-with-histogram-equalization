import utils
import config
import cv2

if __name__ == "__main__":
    Image_Path = config.Image_Path
    Imag = cv2.imread(Image_Path)
    image = utils.cv_image(Imag)
    image.convert2gray()
    image.gamma_correction_image(gamma=0.4)

    raw_hist = image.plt_histogram(img=image.image, name="RAW_HIST", save=True)
    gamma_hist = image.plt_histogram(img=image.new_image, name="GAMMA_HIST", save=True)

    q_k = config.q_k
    q_o = config.q_o
    img = image.histogram_equalization(q_o=q_o, q_k=q_k, hist=gamma_hist[0])

    result_hist = image.plt_histogram(img=img, name="Result_HIST", save=True)

    cv2.imwrite("Result.png", img)
    cv2.imwrite("Gamma_image.png", image.new_image)