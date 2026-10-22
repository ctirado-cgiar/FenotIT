import { Graphviz } from "@hpcc-js/wasm-graphviz"; import fs from "fs";
const g = await Graphviz.load(); fs.writeFileSync(process.argv[3], g.layout(fs.readFileSync(process.argv[2],"utf8"),"svg","dot"));
