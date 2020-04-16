import { Injectable, OnInit } from '@angular/core';
// import { SocketioService } from './socketio.service';
import { Observable, fromEvent } from 'rxjs';
import { Socket } from 'ngx-socket-io';

@Injectable({
  providedIn: 'root'
})
export class DebugService {
  msgs: any[];

  constructor(private socket: Socket) {
    // this.socket.getMessages('log').subscribe(x => this.msgs.push(x));
  }
  getLogs(): Observable<any> {
    return this.socket.fromEvent('log');
  }
}
