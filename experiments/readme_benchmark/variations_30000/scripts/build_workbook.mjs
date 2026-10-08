import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile, FileBlob} from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'outputs');
if(process.argv.includes('--repair-purpose-counts')){
  const finalPath=path.join(out,'FAQ_3000쌍_각10개_변형질문_30000행.xlsx');
  const saved=await SpreadsheetFile.importXlsx(await FileBlob.load(finalPath));
  const catalog=JSON.parse(await fs.readFile(path.join(root,'data','purpose_catalog.json'),'utf8'));
  const manifest=JSON.parse(await fs.readFile(path.join(root,'data','manifest.json'),'utf8'));
  const sheet=saved.worksheets.getItem('검증목적');
  // COUNTIFS wildcard matching is unsupported in this compiler. These are
  // frozen dataset metadata imported from its manifest, not live edit counts.
  sheet.getRange('E5:F5').values=[['주변형 행(생성 시)','태그 포함 행(생성 시)']];
  sheet.getRange('E5:F5').format.rowHeightPx=50;
  sheet.getRange('A3').values=[['생성 시 고정 집계: data/manifest.json. 태그는 여러 개일 수 있어 합계가 질문 수보다 큽니다. REAL_FAILURE는 로그가 없어 0행입니다.']];
  sheet.getRange('E6:F21').clear({applyTo:'contents'});
  sheet.getRange('E6:F21').values=catalog.map(c=>[manifest.primary_variation_counts[c.tag]||0,manifest.validation_tag_counts[c.tag]||0]);
  saved.recalculate();
  const check=await saved.inspect({kind:'table',range:'검증목적!A5:F21',include:'values,formulas',tableMaxRows:17,tableMaxCols:6,maxChars:4000});
  await fs.writeFile(path.join(out,'purpose_counts_inspection.ndjson'),check.ndjson);
  const preview=await saved.render({sheetName:'검증목적',range:'A5:F10',scale:1,format:'png'});
  await fs.writeFile(path.join(out,'purpose_preview.png'),new Uint8Array(await preview.arrayBuffer()));
  const fixed=await SpreadsheetFile.exportXlsx(saved);await fixed.save(finalPath);
  console.log(JSON.stringify({stage:'purpose_counts_repaired',path:finalPath}));
  process.exit(0);
}
const readRows=async name=>(await fs.readFile(path.join(root,'data',name),'utf8')).trim().split('\n').map(JSON.parse);
const [rows,sources,catalog,manifest]=await Promise.all([
  readRows('variations.jsonl'),readRows('source_pairs.jsonl'),
  fs.readFile(path.join(root,'data','purpose_catalog.json'),'utf8').then(JSON.parse),
  fs.readFile(path.join(root,'data','manifest.json'),'utf8').then(JSON.parse)
]);
const validation=JSON.parse(await fs.readFile(path.join(out,'validation.json'),'utf8'));
if(!validation.passed||rows.length!==30000||sources.length!==3000)throw Error('Dataset validation must pass before export');
const wb=Workbook.create();
const main=wb.worksheets.add('변형30000');
const purposes=wb.worksheets.add('검증목적');
const original=wb.worksheets.add('원본3000');
const palette={ink:'#243248',navy:'#203B60',light:'#EDF2F8',muted:'#56677E',amber:'#9A6700'};
function literal(value){return typeof value==='string'&&value.startsWith('=')?"'"+value:value;}
function base(sheet,range){
  sheet.showGridLines=false;
  const area=sheet.getRange(range);
  area.format.font={name:'Arial',size:10,color:palette.ink};
  area.format.verticalAlignment='center';
  area.format.wrapText=true;
}
function header(sheet,address){
  sheet.getRange(address).format={fill:palette.navy,font:{name:'Arial',size:10,color:'#FFFFFF',bold:true},
    horizontalAlignment:'center',verticalAlignment:'center',wrapText:true,rowHeightPx:34};
}
function title(sheet,cell,text){
  sheet.getRange(cell).values=[[text]];
  sheet.getRange(cell).format.font={name:'Arial',size:14,color:palette.navy,bold:true};
  sheet.getRange(cell).format.wrapText=false;
  sheet.getRange(cell).format.rowHeightPx=30;
}
function widths(sheet,widthValues,lastRow){
  widthValues.forEach((width,i)=>sheet.getRangeByIndexes(0,i,lastRow,1).format.columnWidthPx=width);
}
function wrappedLines(text,width){
  const capacity=width-20;
  let lines=0;
  for(const paragraph of String(text??'').split('\n')){
    let x=0,n=1;
    for(const ch of paragraph){const w=/[\uAC00-\uD7AF\u4E00-\u9FFF]/u.test(ch)?13.5:6.9;if(x+w>capacity){n++;x=0;}x+=w;}
    lines+=n;
  }
  return lines;
}
function heights(sheet,matrix,widthValues,startRow){
  const vals=matrix.map(row=>Math.min(545,Math.max(54,Math.ceil((Math.max(...row.map((v,i)=>wrappedLines(v,widthValues[i])))*16+20)/8)*8)));
  for(let i=0;i<vals.length;){let j=i+1;while(j<vals.length&&vals[j]===vals[i])j++;sheet.getRangeByIndexes(startRow+i,0,j-i,widthValues.length).format.rowHeightPx=vals[i];i=j;}
}
function changes(row){
  return row.change_trace.map(c=>{
    if(c.operation==='thousands_separator')return '숫자 천단위 쉼표 제거';
    if(c.operation==='unit_spelling')return '동일 단위를 한글로 표기';
    if(c.operation==='irrelevant_background')return '긴 배경, 질문 위치: '+c.core_position;
    if(c.operation==='move_query_attack_to_document')return '입력 공격 문구를 시험용 문서로 이동';
    return c.from!==undefined?`${c.from} = ${c.to}`:(c.detail||c.operation);
  }).join('; ');
}
const matrix=rows.map(r=>[
  r.parent_case_id,r.case_id,r.type,r.primary_variation,r.query,r.expected_answer,
  r.validation_tags.join(', '),changes(r),r.expected_generation_status,r.source_ids.join(', '),
  r.validation_purposes.join('; '),r.language_pairs.map(t=>`${t.ko} = ${t.en}`).join('; '),
  r.context_fixture_id||'',r.split_group_id,r.type==='RT'?r.repeat_index:'', '독립 검수 필요'
].map(literal));
const last=rows.length+8;
const mainWidths=[115,170,65,190,520,440,260,360,110,155,390,260,280,240,80,155];
base(main,`A1:P${last}`);widths(main,mainWidths,last);main.tabColor=palette.navy;
title(main,'A2','UBot 질문 변형 30,000행');
main.getRange('A4:F4').values=[['원본 쌍',null,'변형/쌍',10,'변형 질문',null]];
main.getRange('B4').formulas=[["=COUNTA('원본3000'!A5:A3004)"]];
main.getRange('F4').formulas=[['=COUNTA(A9:A30008)']];
main.getRange('B4:F4').setNumberFormat('#,##0');
main.getRange('A5').values=[['각 원본당 서로 다른 10개. 원본 정답·조건 상속. 독립 의미·정답 검수는 미완료.']];
main.getRange('A5').format.wrapText=false;
main.getRange('A6').values=[['원본 분류 15개를 각각 2,000행 유지. 반복 입력과 원본 간 같은 질문은 별도 실행 행으로 보존.']];
main.getRange('A6').format.wrapText=false;
main.getRange('A8:P8').values=[['원본 ID','변형 ID','분류','주변형','변형 질문','정답 예시','검증 태그','변경 요소','기대 상태','근거 FAQ','검증 목적','한영 대응어','문서 공격 fixture','원본 묶음','반복 회차','검수 상태']];
header(main,'A8:P8');
for(let start=0;start<matrix.length;start+=1000){main.getRangeByIndexes(start+8,0,Math.min(1000,matrix.length-start),16).values=matrix.slice(start,start+1000);if(start%10000===0)console.log(JSON.stringify({stage:'write_variants',rows:start+1000}));}
heights(main,matrix,mainWidths,8);
main.getRange(`E9:H${last}`).format.verticalAlignment='top';
main.getRange(`K9:L${last}`).format.verticalAlignment='top';
main.getRange(`P9:P${last}`).format.font.color=palette.amber;
main.getRange(`O9:O${last}`).setNumberFormat('0');
const table=main.tables.add(`A8:P${last}`,true,'Variations30000');table.style='TableStyleMedium2';table.showFilterButton=true;
main.freezePanes.freezeRows(8);main.freezePanes.freezeColumns(2);

base(purposes,'A1:F23');widths(purposes,[200,420,340,390,110,110],23);
title(purposes,'A2','검증 목적과 변형 범위');
purposes.getRange('A3').values=[['생성 시 고정 집계: data/manifest.json. 태그는 여러 개일 수 있어 합계가 질문 수보다 큽니다. REAL_FAILURE는 로그가 없어 0행입니다.']];
purposes.getRange('A3').format.wrapText=false;
purposes.getRange('A5:F5').values=[['태그','검증할 실패','보존할 요소','이번 생성 범위','주변형 행(생성 시)','태그 포함 행(생성 시)']];header(purposes,'A5:F5');purposes.getRange('E5:F5').format.rowHeightPx=50;
const purposeMatrix=catalog.map(c=>[c.tag,c.purpose,c.preserve,c.scope,manifest.primary_variation_counts[c.tag]||0,manifest.validation_tag_counts[c.tag]||0]);
purposes.getRange('A6:F21').values=purposeMatrix;
purposes.getRange('E6:F21').setNumberFormat('#,##0');heights(purposes,purposeMatrix,[200,420,340,390,110,110],5);
purposes.freezePanes.freezeRows(5);

const sourceMatrix=sources.map(r=>[r['실행 ID'],r['항목 코드'],r['사용자 질문'],r['정답 예시'],r['기대 상태'],r['근거 원문 FAQ ID (추적용)']||'',r['기대 처리 경로'],r['시험용 변형·가상 데이터']||'',r['필수 사실·표현 기준']].map(literal));
const sourceWidths=[160,65,520,480,110,180,220,420,420];
base(original,'A1:I3004');widths(original,sourceWidths,3004);
title(original,'A1','원본 질문·정답 3,000쌍');
original.getRange('A2').values=[['출처: '+path.basename(manifest.source_workbook)+' | SHA256: '+manifest.source_workbook_sha256]];
original.getRange('A2').format.wrapText=false;
original.getRange('A3').values=[['Context·대화·사용자/API·페르소나 전체 원문은 source_pairs.jsonl에서 원본 ID로 연결합니다. 원본 파일은 수정하지 않았습니다.']];
original.getRange('A3').format.wrapText=false;
original.getRange('A4:I4').values=[['원본 ID','분류','원본 질문','원본 정답 예시','기대 상태','근거 FAQ','처리 경로','가상 시험 데이터','필수 사실·표현']];header(original,'A4:I4');
for(let start=0;start<sourceMatrix.length;start+=1000)original.getRangeByIndexes(start+4,0,Math.min(1000,sourceMatrix.length-start),9).values=sourceMatrix.slice(start,start+1000);
heights(original,sourceMatrix,sourceWidths,4);original.getRange('C5:D3004').format.verticalAlignment='top';
const sourceTable=original.tables.add('A4:I3004',true,'OriginalPairs3000');sourceTable.style='TableStyleMedium2';
original.freezePanes.freezeRows(4);original.freezePanes.freezeColumns(2);

console.log(JSON.stringify({stage:'recalculate'}));
wb.recalculate();
const inspection=await wb.inspect({kind:'table',range:'변형30000!A2:F10',include:'values,formulas',tableMaxRows:10,tableMaxCols:6,maxChars:2500});
await fs.writeFile(path.join(out,'workbook_inspection.ndjson'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'formula_error_scan',maxChars:2000});
await fs.writeFile(path.join(out,'workbook_formula_scan.ndjson'),errors.ndjson);
for(const [name,sheetName,range] of [
 ['questions_preview','변형30000','A8:F12'],
 ['purpose_preview','검증목적','A5:F10'],
 ['source_preview','원본3000','A4:F8'],
 ['long_question_preview','변형30000','D17:F18']
]){
  console.log(JSON.stringify({stage:'render',name}));
  const image=await wb.render({sheetName,range,scale:1,format:'png'});
  await fs.writeFile(path.join(out,name+'.png'),new Uint8Array(await image.arrayBuffer()));
}
console.log(JSON.stringify({stage:'export'}));
const xlsx=await SpreadsheetFile.exportXlsx(wb);
const finalPath=path.join(out,'FAQ_3000쌍_각10개_변형질문_30000행.xlsx');
await xlsx.save(finalPath);
await fs.writeFile(path.join(out,'workbook_export.json'),JSON.stringify({path:finalPath,variant_rows:rows.length,source_rows:sources.length,sheets:['변형30000','검증목적','원본3000'],rendered:true},null,2)+'\n');
console.log(JSON.stringify({stage:'saved',path:finalPath}));
