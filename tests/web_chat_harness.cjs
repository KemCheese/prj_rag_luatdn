// Execute the actual embedded script with deterministic DOM/fetch substitutes.
const fs = require("node:fs");
const vm = require("node:vm");
const input = JSON.parse(fs.readFileSync(0, "utf8"));

class Element {
    constructor() {
        this.html = "";
        this.text = "";
        this.disabled = false;
        this.focused = false;
    }
    get innerHTML() { return this.html; }
    set innerHTML(value) { this.html = value; this.text = ""; }
    get innerText() { return this.text; }
    set innerText(value) { this.text = value; this.html = ""; }
    focus() { this.focused = true; }
}

async function main() {
    let bot;
    const elements = {q: new Element(), btn: new Element(), msgs: new Element()};
    elements.q.value = "Câu hỏi hợp lệ?";
    elements.msgs.appendChild = element => { bot = element; };
    const status = input.status ?? 200;
    const context = {
        document: {
            getElementById: id => elements[id],
            createElement: () => new Element(),
        },
        fetch: async () => {
            if (input.networkError) throw new Error(input.networkError);
            return {
                ok: status >= 200 && status < 300,
                status,
                json: async () => {
                    if (input.invalidJson) throw new SyntaxError("Invalid JSON body");
                    return input.data;
                },
            };
        },
    };
    vm.createContext(context);
    vm.runInContext(input.script, context, {timeout: 1000});
    await context.send();
    process.stdout.write(JSON.stringify({
        html: bot.innerHTML,
        text: bot.innerText,
        inputEnabled: !elements.q.disabled,
        buttonEnabled: !elements.btn.disabled,
        focused: elements.q.focused,
    }));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
