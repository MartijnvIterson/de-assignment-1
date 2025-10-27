document.getElementById('uploadForm').addEventListener('submit', async function(e){
  e.preventDefault();
  const form = e.currentTarget;
  const fileInput = document.getElementById('file');
  if(!fileInput.files.length){ alert('Please choose a CSV file'); return }
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  formData.append('project_id', document.getElementById('project_id').value);
  formData.append('model_gcs', document.getElementById('model_gcs').value);

  const res = await fetch('/predict', { method: 'POST', body: formData });
  if(!res.ok){
    const err = await res.json().catch(()=>({error:'unknown'}));
    alert('Predict failed: '+(err.error||res.statusText));
    return;
  }
  const data = await res.json();
  const results = data.results || [];
  const container = document.getElementById('results');
  if(!results.length){ container.innerHTML = '<div class="small">No rows returned</div>'; return }

  // build table
  let html = '<div class="table-wrapper"><table class="table"><thead><tr>';
  const keys = Object.keys(results[0]);
  for(const k of keys){ html += `<th>${k}</th>` }
  html += '</tr></thead><tbody>';
  for(const r of results){ html += '<tr>'; for(const k of keys){
    if(k==='prediction'){
      html += `<td><span class="pred-chip pred-${r[k]}">${r[k]===1? 'Fraud':'OK'}</span></td>`
    } else {
      html += `<td>${String(r[k]).slice(0,80)}</td>`
    }
  } html += '</tr>' }
  html += '</tbody></table></div>';
  container.innerHTML = html + '<div class="footer">Predicted using model from: '+document.getElementById('model_gcs').value+'</div>';
});
