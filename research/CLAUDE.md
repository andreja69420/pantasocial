# research/ - instructions for Claude

## Radna pravila (korisnik, obavezno za sve)
- Sve što može da ide paralelno ide paralelno: scrapovi više kategorija/sajtova istovremeno, treninzi na svim slobodnim jezgrima, evaluacije dok traju drugi poslovi. Nikad ne čekaj sekvencijalno ako posao ne zavisi od prethodnog.
- Brzina scrapovanja: kad je poznata brzina koja radi (Flashscore ~150/s), drži je fiksno. Spor odgovor (>5 s) = samo malo smanji brzinu, BEZ pauze, pa je vrati kad prođe. Pauza samo na pravu grešku (429/5xx/pad veze). Na novom sajtu: kreni umereno i diži dok ne krenu greške, pa se vrati na poslednju brzinu koja je radila.
- Rezultate upisuj u README, commit i push posle svakog koraka.
- Izuzetak zbog memorije (15 GB): pun `features2.py` preračun uzima ~8 GB, pa najviše jedan istovremeno; sve ostalo paralelno.
- Rupe i greške u podacima (mečevi koji fale, duplikati, pogrešni igrači ili kola): odmah dopuni ili ispravi izvor, ispravi kod da se ne ponovi, ponovo izgradi features (`features2.py`, petlja stoji dok traje gradnja, pa zamena `out_live2`), proveri novu bazu prema staroj, upiši u README, commit i push. Ne čekaj odobrenje.
