export class Log {
    constructor(dict: any) {
        this.time = new Date(dict.time);
    }

    time: Date;

    static fromJson(jsondict: any[]): Log[] {
        return jsondict.map((x: any) => new Log(x));
    }

}
