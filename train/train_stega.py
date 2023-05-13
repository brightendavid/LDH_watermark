from model_stega import stega_net
import torch

Encoder = stega_net.StegaStampEncoder().to("cuda")  # stega
Decoder = stega_net.StegaStampDecoder().to("cuda")  # stega
Discriminator = stega_net.StegaStampDiscriminator().to("cuda")

img = torch.randn((8, 3, 400, 400)).to("cuda")
msg = torch.randn(8, 100).to("cuda")

res_img = Encoder({"img": img, "msg": msg})
msg_pred, stn_img = Decoder(img)
dis_pred = Discriminator(img)

if __name__ == '__main__':
    pass
