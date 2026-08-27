# Probă fizică controlată — autonomie G1 v20

## Pornire

```bash
cd /home/unitree/dashboard_g1_v20
bash start_dashboard.sh
```

Scriptul pornește backend-ul și `send_video_depth.py`. Nu porni încă o instanță
separată a camerei. Repornește dashboard-ul după orice modificare de backend.

## Pregătire

1. Eliberează o zonă de cel puțin 1,5 m în jurul robotului și ține STOP-ul la îndemână.
2. Încarcă harta folosită de SLAM (de exemplu `harta_buna_2707.pcd`).
3. Setează poziția și orientarea inițială pe hartă (API 1804).
4. Așteaptă ca poziția afișată să fie stabilă și diferită de o valoare implicită falsă.
5. Previzualizează o țintă aflată la aproximativ 1 m, cu viteza `0.15 m/s` și timeout `0` (fără limită de timp).
6. Confirmă numai dacă ruta desenată nu intersectează zona portocalie/roșie.

Butonul de confirmare rulează acum preflight-ul. Navigarea este limitată la `0,22 m/s`, iar pornirea la `0,15 m/s`. Navigarea este limitată la `0,22 m/s`, iar pornirea la `0,15 m/s`. Toate acestea trebuie să fie active:
harta, modul localization, poziția Unitree, telemetria SLAM, RealSense depth recent și
cloud LiDAR SLAM recent. Autonomia nu pornește doar cu unul dintre cei doi senzori.

## Probe în ordine

1. **Liber:** A→B la 1 m, fără obstacol. Rezultat acceptat: ajunge, nu oscilează și se oprește la țintă.
2. **Colț lateral static:** pune colțul mesei în jumătatea stângă/dreaptă a camerei, la circa 0,9 m. Rezultat acceptat: camera oprește preventiv, iar ocolirea este calculată numai dacă LiDAR-ul confirmă geometria.
3. **Obstacol apărut dinamic:** introdu un obiect pe rută. Rezultat acceptat: STOP, confirmare LiDAR, invalidarea rutei vechi, validarea noii rute timp de 0,30 s și ocolire automată.
4. **Culoar:** plannerul încearcă profilul sigur de `0,25 m`; dacă nu există rută, folosește automat profilul controlat de `0,20 m` și reduce viteza. Raza sub `0,20 m` este refuzată.
5. **Recuperare laterală:** este dezactivată implicit. Se testează separat numai după validarea replanificării normale și se activează explicit cu `G1_ENABLE_LATERAL_RECOVERY=1`.
6. **Coadă:** în timpul primei deplasări adaugă al doilea punct. Rezultat acceptat: este executat numai după sosirea la primul.

După fiecare probă apasă **Descarcă jurnalul cursei**. Fișierul JSONL conține
comenzile 1102, latența feedback-ului, opririle 1201, reluările 1202, poziția,
starea celor doi senzori și fiecare rută recalculată.

## Oprire imediată

La orice abatere apasă `STOP navigare`. Nu continua proba dacă ruta sare în alt frame,
poziția SLAM se blochează, fluxurile senzorilor devin vechi sau mâinile se apropie de
mobilier mai mult decât permite anvelopa de siguranță.

## Criteriu de acceptare

Considerăm versiunea validată fizic numai după trei repetări consecutive ale probelor
1–3, fără contact, fără pornirea inițială neconfirmată și fără pierderea localizării.

Fiecare cursă confirmată creează automat un fișier `logs/nav_*.jsonl`. Calea apare în
interfață după pornire. Ultimele evenimente pot fi citite și din
`GET /api/nav/flight_log`; jurnalul conține starea navigatorului, poziția, senzorii și
telemetria SLAM necesare pentru analiza unei opriri sau abateri.
