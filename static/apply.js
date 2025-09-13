/* apply.js
   - Upload Patta to Python backend for OCR
   - Multi-state language support
   - Auto-fill form fields
   - Persistent map with manual selection
   - Submit to backend
*/

(async function () {
  const form = document.getElementById('applyForm');
  const statusEl = document.getElementById('status');
  const resultBox = document.getElementById('result');
  const matchInfo = document.getElementById('matchInfo');
  const confirmBtn = document.getElementById('confirmBtn');
  const chooseManual = document.getElementById('chooseManual');
  const manualHint = document.getElementById('manualHint');
  const resetSelection = document.getElementById('resetSelection');

  const pattaFileInput = document.getElementById('pattaFile');
  const extractBtn = document.getElementById('extractBtn');
  const extractedInfo = document.getElementById('extractedInfo');

  let map = null;
  let manualMarker = null;
  let highlightLayer = null;
  let manualMode = false;
  let manualCoord = null;
  let loadedStateGeo = null;
  let lastMatch = null;

  // ------------------- Map -------------------
  function initMap() {
    if (map) return;
    map = L.map('map').setView([22.0, 82.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

    map.on('click', async function (e) {
      if (!manualMode) return;
      manualCoord = [e.latlng.lng, e.latlng.lat];
      if (manualMarker) map.removeLayer(manualMarker);
      manualMarker = L.marker([manualCoord[1], manualCoord[0]]).addTo(map)
        .bindPopup('Selected location').openPopup();

      manualHint.textContent = `Selected: ${manualCoord[1].toFixed(6)}, ${manualCoord[0].toFixed(6)} — finding nearest feature...`;

      const stateKey = document.getElementById('stateKey').value;
      if (!stateKey) { statusEl.textContent = 'Select a state first.'; return; }

      try {
        const geo = loadedStateGeo && loadedStateGeo._stateKey === stateKey
          ? loadedStateGeo : await loadStateGeo(stateKey);
        const found = findNearestAvailableFeature(geo, manualCoord[0], manualCoord[1]);
        if (!found) {
          statusEl.textContent = 'No available features near selected spot.';
          return;
        }
        lastMatch = { stateKey, feature: found.feature, distanceMeters: found.distMeters, villageCoord: manualCoord };
        showMatchToUser(lastMatch.feature, lastMatch.distanceMeters, lastMatch.villageCoord);
      } catch (err) {
        console.error(err);
        statusEl.textContent = 'Error matching feature: ' + err.message;
      } finally {
        manualMode = false;
        manualHint.textContent = '';
      }
    });
  }

  // ------------------- GeoJSON -------------------
  function candidateUrlsFor(stateKey) {
    return [`/static/geojson/${stateKey}.geojson`];
  }

  async function fetchWithFallback(urls) {
    for (const u of urls) {
      try {
        const res = await fetch(u);
        if (!res.ok) continue;
        return { url: u, json: await res.json() };
      } catch (e) { continue; }
    }
    return null;
  }

  async function loadStateGeo(stateKey) {
    statusEl.textContent = 'Loading state features...';
    const resp = await fetchWithFallback(candidateUrlsFor(stateKey));
    if (!resp) throw new Error('State geojson not found');
    loadedStateGeo = resp.json;
    loadedStateGeo._stateKey = stateKey;
    return loadedStateGeo;
  }

  function validFeatureForMatching(feature) {
    if (!feature || !feature.geometry) return false;
    const apps = feature.properties?.applications;
    if (Array.isArray(apps) && apps.length > 0) return false;
    return ["Point", "Polygon", "MultiPolygon"].includes(feature.geometry.type);
  }

  function findNearestAvailableFeature(geojson, lon, lat) {
    const p = turf.point([lon, lat]);
    let best = null;
    for (const f of geojson.features || []) {
      if (!validFeatureForMatching(f)) continue;
      let pt = (f.geometry.type === 'Point') ? turf.point(f.geometry.coordinates) : turf.centroid(f);
      const distM = turf.distance(p, pt, { units: 'kilometers' }) * 1000;
      if (!best || distM < best.distMeters) best = { feature: f, distMeters: distM };
    }
    return best;
  }

  function showMatchToUser(feature, distanceMeters, villageCoord) {
    resultBox.style.display = 'block';
    const fid = feature.id || feature.properties?.['@id'] || 'unknown';
    matchInfo.innerHTML = `
      Feature ID: <strong>${fid}</strong><br/>
      Name: <em>${feature.properties?.name || '—'}</em><br/>
      Distance: <strong>${Math.round(distanceMeters)} m</strong>
      ${villageCoord ? `<br/>Village coords: ${villageCoord[1].toFixed(5)}, ${villageCoord[0].toFixed(5)}` : ''}
    `;
    if (highlightLayer) map.removeLayer(highlightLayer);
    highlightLayer = L.layerGroup().addTo(map);
    if (feature.geometry.type === 'Point') highlightLayer.addLayer(L.marker([feature.geometry.coordinates[1], feature.geometry.coordinates[0]]));
    if (villageCoord) highlightLayer.addLayer(L.circle([villageCoord[1], villageCoord[0]], { radius: Math.max(40, distanceMeters), color:'#3388ff', weight:1, fill:false }));
    map.setView(villageCoord ? [villageCoord[1], villageCoord[0]] : [feature.geometry.coordinates[1], feature.geometry.coordinates[0]], 14);
  }

  // ------------------- OCR Extraction -------------------
  extractBtn.addEventListener('click', async () => {
    const file = pattaFileInput.files[0];
    const stateKey = document.getElementById('stateKey').value;
    if (!file || !stateKey) { alert('Select state and upload Patta'); return; }

    extractedInfo.textContent = 'Extracting text from Patta...';
    statusEl.textContent = 'Sending Patta to server for OCR...';

    try {
      const formData = new FormData();
      formData.append('pattaFile', file);
      formData.append('stateKey', stateKey);

      const res = await fetch('/extract_patta', { method: 'POST', body: formData });
      if (!res.ok) throw new Error('OCR backend failed');

      const data = await res.json();
      const { name, village } = data;

      extractedInfo.innerHTML = `✅ Extracted Info: <br/>Name: <b>${name}</b><br/>Village: <b>${village}</b><br/>State: <b>${stateKey}</b>`;

      document.getElementById('applicant').value = name;
      document.getElementById('village').value = village;
      document.getElementById('stateKey').value = stateKey;

      statusEl.textContent = 'OCR completed, fields auto-filled.';

    } catch (err) {
      console.error(err);
      statusEl.textContent = 'OCR failed: ' + err.message;
    }
  });

  // ------------------- Form submit -------------------
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    resultBox.style.display = 'none';
    statusEl.textContent = '';
    lastMatch = null;

    const applicant = document.getElementById('applicant').value.trim();
    const tribal_note = document.getElementById('tribal_note').value.trim();
    const village = document.getElementById('village').value.trim();
    const stateKey = document.getElementById('stateKey').value;

    if (!applicant || !stateKey) { statusEl.textContent = 'Fill required fields (name and state).'; return; }

    try {
      let coords = manualCoord || await geocodeVillage(village, stateKey);
      if (!coords) { statusEl.textContent = 'Enter village or pick map point'; return; }

      if (!loadedStateGeo || loadedStateGeo._stateKey !== stateKey) await loadStateGeo(stateKey);
      const found = findNearestAvailableFeature(loadedStateGeo, coords[0], coords[1]);
      if (!found) { statusEl.textContent = 'No available features found'; return; }

      lastMatch = { stateKey, feature: found.feature, distanceMeters: found.distMeters, villageCoord: coords, applicant, tribal_note };
      showMatchToUser(found.feature, found.distMeters, coords);

    } catch (err) { console.error(err); statusEl.textContent = 'Error: ' + err.message; }
  });

  // ------------------- Manual selection -------------------
  chooseManual.addEventListener('click', () => {
    initMap();
    manualMode = true;
    manualHint.textContent = 'Click map to select a spot (nearest feature will be matched).';
    statusEl.textContent = 'Manual selection enabled';
  });

  resetSelection.addEventListener('click', () => {
    manualCoord = null;
    if (manualMarker) { map.removeLayer(manualMarker); manualMarker = null; }
    if (highlightLayer) { map.removeLayer(highlightLayer); highlightLayer = null; }
    resultBox.style.display = 'none';
    statusEl.textContent = 'Selection cleared.';
  });

  // ------------------- Submit to backend -------------------
  async function submitApplicationToBackend(stateKey, featureId, appObj) {
    try {
      const res = await fetch('/save_app', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stateKey, featureId, app: appObj })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return await res.json();
    } catch (err) {
      console.error(err);
      throw err;
    }
  }

  confirmBtn.addEventListener('click', async () => {
    if (!lastMatch) { statusEl.textContent = 'No match to submit'; return; }

    const appObj = {
      id: 'app-' + Date.now(),
      applicant: lastMatch.applicant || document.getElementById('applicant').value.trim(),
      tribal_note: lastMatch.tribal_note || document.getElementById('tribal_note').value.trim(),
      coords: lastMatch.villageCoord || lastMatch.feature.geometry?.coordinates || null,
      status: 'pending',
      created_at: new Date().toISOString(),
      admin_note: ''
    };

    statusEl.textContent = 'Submitting application...';

    try {
      await submitApplicationToBackend(lastMatch.stateKey, lastMatch.feature.id, appObj);
      statusEl.textContent = 'Application submitted successfully';

      // Reflect locally so same feature won't be picked again
      lastMatch.feature.properties = lastMatch.feature.properties || {};
      lastMatch.feature.properties.applications = lastMatch.feature.properties.applications || [];
      lastMatch.feature.properties.applications.push(appObj);

    } catch (err) {
      console.error(err);
      statusEl.textContent = 'Failed to submit: ' + err.message;
    }
  });

  // ------------------- Map initialization -------------------
  initMap();
})();

