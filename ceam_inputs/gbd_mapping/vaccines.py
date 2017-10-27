from .templates import Vaccine, Vaccines, meid

vaccines = Vaccines(
    rota_virus=Vaccine(
        name='rota_virus',
        gbd_id=meid(10596),
    ),
    dtp3=Vaccine(
        name='dtp3',
        gbd_id=None,
    ),
)
