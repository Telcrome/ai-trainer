import { Injectable } from '@angular/core';

interface ServerCreds {
  url: string;
}

@Injectable({
  providedIn: 'root'
})
export class ConfigService {

  private _serverCreds: ServerCreds;
  public get serverCreds(): ServerCreds {
    if (!!this._serverCreds) {
      this._serverCreds = JSON.parse(window.localStorage.getItem('config'));
    }
    return this._serverCreds;
  }
  public set serverCreds(v: ServerCreds) {
    this._serverCreds = v;
    window.localStorage.setItem('config', JSON.stringify(v));
  }

  reset() {
    window.localStorage.clear();
  }
}
