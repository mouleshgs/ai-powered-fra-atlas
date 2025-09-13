/* Interactive apply script:
   - persistent map under the form
   - "Choose different spot" lets user click map to select village coords
   - loads state geojson (tries /templates/, /static/, relative)
   - finds nearest feature that has no existing applications
   - allows confirm -> POST to backend /save_app
*/

(function(){
  const form = document.getElementById('applyForm');
  const statusEl = document.getElementById('status');
  const resultBox = document.getElementById('result');
  const matchInfo = document.getElementById('matchInfo');
  const confirmBtn = document.getElementById('confirmBtn');
  const chooseManual = document.getElementById('chooseManual');
  const manualHint = document.getElementById('manualHint');
  const resetSelection = document.getElementById('resetSelection');

  let map = null;
  let manualMarker = null;
  let highlightLayer = null;
  let manualMode = false;
  let manualCoord = null; // [lon, lat]
  let loadedStateGeo = null;
  let lastMatch = null; // { stateKey, feature, distanceMeters, villageCoord, applicant, tribal_note }

  function candidateUrlsFor(stateKey){
    return [
      `/static/geojson/${stateKey}.geojson`
    ];
  }

  async function fetchWithFallback(urls){
    for(const u of urls){
      try {
        const res = await fetch(u);
        if(!res.ok) continue;
        const j = await res.json();
        return { url: u, json: j };
      } catch(e){
        continue;
      }
    }
    return null;
  }

async function geocodeVillage(village, state){
  const q = encodeURIComponent(`${village} ${state}`);
  const url = `/geocode?q=${q}`;
  statusEl.textContent = 'Geocoding village via backend...';
  const res = await fetch(url);
  if(!res.ok) throw new Error('Geocode failed: ' + res.status);
  const arr = await res.json();
  if(!arr || arr.length === 0) return null;
  const item = arr[0];
  return { lat: parseFloat(item.lat), lon: parseFloat(item.lon), display_name: item.display_name };
}


  async function loadStateGeo(stateKey){
    statusEl.textContent = 'Loading state features...';
    const cand = candidateUrlsFor(stateKey);
    const resp = await fetchWithFallback(cand);
    if(!resp) throw new Error('State geojson not found. Tried: ' + cand.join(', '));
    loadedStateGeo = resp.json;
    return loadedStateGeo;
  }

  function validFeatureForMatching(feature){
    if(!feature || !feature.geometry) return false;
    const g = feature.geometry;
    if(g.type === 'Point' && Array.isArray(g.coordinates) && typeof g.coordinates[0] === 'number' && typeof g.coordinates[1] === 'number') {
      const apps = feature.properties && feature.properties.applications;
      if(Array.isArray(apps) && apps.length>0) return false;
      return true;
    }
    if((g.type === 'Polygon' || g.type === 'MultiPolygon') && feature.properties){
      const apps = feature.properties.applications;
      if(Array.isArray(apps) && apps.length>0) return false;
      return true;
    }
    return false;
  }

  function findNearestAvailableFeature(geojson, lon, lat){
    const p = turf.point([lon, lat]);
    let best = null;
    for(const f of (geojson.features||[])){
      if(!validFeatureForMatching(f)) continue;
      let pt;
      if(f.geometry.type === 'Point') pt = turf.point(f.geometry.coordinates, { _id: f.id });
      else {
        try { pt = turf.centroid(f); } catch(e){ continue; }
        pt.properties = { _id: f.id };
      }
      const distKm = turf.distance(p, pt, { units: 'kilometers' });
      const distM = distKm * 1000;
      if(best === null || distM < best.distMeters){
        best = { feature: f, distMeters: distM };
      }
    }
    return best;
  }

  async function submitApplicationToBackend(stateKey, featureId, appObj){
    const url = 'http://localhost:5000/save_app';
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stateKey, featureId, appId: appObj.id, updatedApp: appObj })
    });
    const json = await (res.headers.get('content-type') && res.headers.get('content-type').includes('application/json') ? res.json() : null);
    if(!res.ok) throw new Error((json && json.message) ? json.message : `HTTP ${res.status}`);
    return json;
  }

  // map init
  function initMap(){
    if(map) return;
    map = L.map('map').setView([22.0,82.0], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{ maxZoom:18 }).addTo(map);

    // click handler for manual selection
    map.on('click', async function(e){
      if(!manualMode) return;
      manualCoord = [ e.lngLat ? e.lngLat.lng : (e.latlng && e.latlng.lng), e.latlng ? e.latlng.lat : (e.latlng && e.latlng.lat) ];
      if(manualMarker) map.removeLayer(manualMarker);
      manualMarker = L.marker([manualCoord[1], manualCoord[0]]).addTo(map).bindPopup('Selected location').openPopup();
      manualHint.textContent = `Selected: ${manualCoord[1].toFixed(6)}, ${manualCoord[0].toFixed(6)} — finding nearest available feature...`;
      // require a state to be selected
      const stateKey = document.getElementById('stateKey').value;
      if(!stateKey){ statusEl.textContent = 'Select a state first.'; return; }
      try {
        const geo = loadedStateGeo && loadedStateGeo._stateKey === stateKey ? loadedStateGeo : await loadStateGeo(stateKey);
        if(geo) geo._stateKey = stateKey;
        const found = findNearestAvailableFeature(geo, manualCoord[0], manualCoord[1]);
        if(!found){
          statusEl.textContent = 'No available features without applications found near selected spot.';
          return;
        }
        lastMatch = { stateKey, feature: found.feature, distanceMeters: found.distMeters, villageCoord: manualCoord };
        showMatchToUser(lastMatch.feature, lastMatch.distanceMeters, lastMatch.villageCoord);
      } catch(err){
        console.error(err);
        statusEl.textContent = 'Error while matching: ' + (err.message || err);
      } finally {
        // exit manual mode after a selection
        manualMode = false;
        manualHint.textContent = '';
      }
    });
  }

  function showMatchToUser(feature, distanceMeters, villageCoord){
    resultBox.style.display = 'block';
    const fid = feature.id || (feature.properties && feature.properties['@id']) || 'unknown';
    matchInfo.innerHTML = `
      Feature ID: <strong>${escapeHtml(String(fid))}</strong><br/>
      Name: <em>${escapeHtml((feature.properties && (feature.properties.name || feature.properties['@id'])) || '—')}</em><br/>
      Distance: <strong>${Math.round(distanceMeters)} m</strong>
      ${villageCoord ? `<br/>Village coords: ${villageCoord[1].toFixed(5)}, ${villageCoord[0].toFixed(5)}` : ''}
    `;
    statusEl.textContent = 'Verify the matched feature. If correct, confirm to submit application.';
    // highlight on map
    if(highlightLayer) { map.removeLayer(highlightLayer); highlightLayer = null; }
    const coords = feature.geometry && feature.geometry.type === 'Point' ? feature.geometry.coordinates : null;
    const marker = coords ? L.marker([coords[1], coords[0]], { title: 'Matched feature' }) : null;
    highlightLayer = L.layerGroup();
    if(marker) highlightLayer.addLayer(marker);
    // add small circle showing match distance
    if(villageCoord){
      highlightLayer.addLayer(L.circle([villageCoord[1], villageCoord[0]], { radius: Math.max(40, distanceMeters), color:'#3388ff', weight:1, fill:false }));
    }
    highlightLayer.addTo(map);
    // center map on matched feature
    try {
      if(coords) map.setView([coords[1], coords[0]], 14);
      else if(villageCoord) map.setView([villageCoord[1], villageCoord[0]], 14);
    } catch(e){}
  }

  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    resultBox.style.display = 'none';
    statusEl.textContent = '';
    lastMatch = null;

    const applicant = document.getElementById('applicant').value.trim();
    const tribal_note = document.getElementById('tribal_note').value.trim();
    const village = document.getElementById('village').value.trim();
    const stateKey = document.getElementById('stateKey').value;

    if(!applicant || !stateKey){ statusEl.textContent = 'Please fill required fields (name and state).'; return; }

    try {
      let coordsSource = null;
      let coords = null;
      if(manualCoord){
        coordsSource = 'manual';
        coords = manualCoord;
      } else if(village){
        const geoV = await geocodeVillage(village, stateKey);
        if(!geoV){
          statusEl.textContent = 'Village not found by geocoder. You can choose a spot on the map using "Choose different spot".';
          return;
        }
        coordsSource = 'geocode';
        coords = [geoV.lon, geoV.lat];
        statusEl.textContent = `Village located: ${geoV.display_name} (${geoV.lat.toFixed(5)}, ${geoV.lon.toFixed(5)})`;
      } else {
        statusEl.textContent = 'Please enter village OR pick a point on the map (Choose different spot).';
        return;
      }

      // load state geojson if not loaded or different state
      if(!loadedStateGeo || loadedStateGeo._stateKey !== stateKey){
        const geo = await loadStateGeo(stateKey);
        if(geo) geo._stateKey = stateKey;
      }
      const found = findNearestAvailableFeature(loadedStateGeo, coords[0], coords[1]);
      if(!found){
        statusEl.textContent = 'No available mapped features without existing applications were found in this state near provided coords.';
        return;
      }
      lastMatch = { stateKey, feature: found.feature, distanceMeters: found.distMeters, villageCoord: coords, applicant, tribal_note };
      showMatchToUser(lastMatch.feature, lastMatch.distanceMeters, lastMatch.villageCoord);

    } catch(err){
      console.error(err);
      statusEl.textContent = 'Error: ' + (err && err.message ? err.message : String(err));
    }
  });

  chooseManual.addEventListener('click', async () => {
    initMap();
    manualMode = true;
    manualHint.textContent = 'Click on the map to select a spot (then nearest available feature will be matched).';
    statusEl.textContent = 'Manual selection enabled — click map to pick location.';
  });

  confirmBtn.addEventListener('click', async () => {
    if(!lastMatch || !lastMatch.feature){ statusEl.textContent = 'No match to submit.'; return; }
    const id = 'app-user-' + Date.now();
    const appObj = {
      id,
      applicant: lastMatch.applicant || document.getElementById('applicant').value.trim(),
      tribal_note: lastMatch.tribal_note || document.getElementById('tribal_note').value.trim(),
      coords: lastMatch.villageCoord || (lastMatch.feature.geometry && lastMatch.feature.geometry.coordinates) || null,
      status: 'pending',
      created_at: new Date().toISOString(),
      admin_note: ''
    };

    statusEl.textContent = 'Submitting application...';
    try {
      await submitApplicationToBackend(lastMatch.stateKey, String(lastMatch.feature.id), appObj);
      statusEl.textContent = 'Application submitted successfully (pending review).';
      // reflect locally so next immediate search won't pick same feature
      if(!lastMatch.feature.properties) lastMatch.feature.properties = {};
      lastMatch.feature.properties.applications = lastMatch.feature.properties.applications || [];
      lastMatch.feature.properties.applications.push(appObj);
      // hide result optionally
      // resultBox.style.display = 'none';
    } catch(e){
      console.error(e);
      statusEl.textContent = 'Failed to submit: ' + (e && e.message ? e.message : String(e));
    }
  });

  resetSelection.addEventListener('click', () => {
    manualCoord = null;
    if(manualMarker) { map.removeLayer(manualMarker); manualMarker = null; }
    if(highlightLayer) { map.removeLayer(highlightLayer); highlightLayer = null; }
    resultBox.style.display = 'none';
    statusEl.textContent = 'Selection cleared.';
  });

  // initialize map at load
  initMap();

  function escapeHtml(s){ if(!s) return ''; return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
})();
