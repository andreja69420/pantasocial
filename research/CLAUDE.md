# research/ - instructions for Claude

## Radna pravila (korisnik, obavezno za sve)
- Sve što može da ide paralelno ide paralelno: scrapovi više kategorija/sajtova istovremeno, treninzi na svim slobodnim jezgrima, evaluacije dok traju drugi poslovi. Nikad ne čekaj sekvencijalno ako posao ne zavisi od prethodnog.
- Brzina scrapovanja: kreni umereno, pa diži broj zahteva/niti sve dok ne krenu greške ili usporavanje (rate limit). Kad krene: vrati se na poslednju brzinu koja je radila, pauza 1-2 min, probaj opet; ako i dalje ne ide, spuštaj korak po korak (npr. 35 -> 30 -> 25 -> 20). Spori odgovori (>5 s) se računaju kao znak rate limita.
- Rezultate upisuj u README, commit i push posle svakog koraka.
- Izuzetak zbog memorije (15 GB): pun `features2.py` preračun uzima ~8 GB, pa najviše jedan istovremeno; sve ostalo paralelno.
